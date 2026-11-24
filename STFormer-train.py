import os
import gc
import sys
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_spatio_temporal_data(lon, lat, time_days, years, reference_year=2017):
    lon_norm = lon / 180.0
    lat_norm = lat / 90.0
    time_norm = (time_days - 1.0) / 365.25
    year_norm = years - float(reference_year)
    return (lon_norm.astype(np.float32), lat_norm.astype(np.float32),
            time_norm.astype(np.float32), year_norm.astype(np.float32))


DTYPE = torch.float32


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_frequencies=16):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        frequencies = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies, dtype=DTYPE)
        self.register_buffer('frequencies', frequencies)
        self.projection = nn.Linear(num_frequencies * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, lon, lat):
        batch_size = lon.shape[0]
        lon = lon.unsqueeze(1)
        lat = lat.unsqueeze(1)
        lon_enc = torch.cat([
            torch.sin(lon * self.frequencies),
            torch.cos(lon * self.frequencies)
        ], dim=1)
        lat_enc = torch.cat([
            torch.sin(lat * self.frequencies),
            torch.cos(lat * self.frequencies)
        ], dim=1)
        spatial_enc = torch.cat([lon_enc, lat_enc], dim=1)
        spatial_enc = self.projection(spatial_enc)
        return self.norm(spatial_enc)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_frequencies=16):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        frequencies = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies, dtype=DTYPE) * 2 * np.pi
        self.register_buffer('frequencies', frequencies)
        self.time_projection = nn.Linear(num_frequencies * 2, d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.year_projection = nn.Linear(1, d_model)
        self.year_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, time, year):
        time = time.unsqueeze(1)
        year = year.unsqueeze(1)
        time_enc = torch.cat([
            torch.sin(time * self.frequencies),
            torch.cos(time * self.frequencies)
        ], dim=1)
        time_emb = self.time_norm(self.time_projection(time_enc))
        year_emb = self.year_norm(self.year_projection(year))
        combined = torch.cat([time_emb, year_emb], dim=1)
        return self.fusion(combined)


class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, spatial_freq=16, temporal_freq=16):
        super().__init__()
        self.spatial_enc = SpatialPositionalEncoding(d_model, spatial_freq)
        self.temporal_enc = TemporalPositionalEncoding(d_model, temporal_freq)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, lon, lat, time, year):
        spatial_emb = self.spatial_enc(lon, lat)
        temporal_emb = self.temporal_enc(time, year)
        combined = torch.cat([spatial_emb, temporal_emb], dim=1)
        return self.fusion(combined)


class TransformerWithLearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        return x + self.positional_embeddings(positions)


class SimpleMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def estimate_unfold_length(input_length, patch_size, stride):
    return (input_length - patch_size) // stride + 1 if stride > 0 else 0


class MyModel(nn.Module):
    def __init__(self, patch_size, stride, feature_dim, d_model, nhead,
                 num_transformer_layers, dim_feedforward, final_mlp_layers,
                 input_radiance_length, spatial_freq=16, temporal_freq=16):
        super().__init__()
        seq_len = estimate_unfold_length(input_radiance_length, patch_size, stride)
        self.seq_positional_encoder = TransformerWithLearnedPositionalEmbedding(d_model, seq_len + 1)
        self.spatio_temporal_encoder = SpatioTemporalPositionalEncoding(
            d_model, spatial_freq, temporal_freq
        )
        self.radiance_linear = nn.Linear(patch_size, d_model)
        self.uniform_linear = nn.Linear(feature_dim, d_model)
        self.radiance_norm = nn.LayerNorm(d_model)
        self.feature_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True, dtype=DTYPE
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_transformer_layers
        )
        self.final_nn = SimpleMLP([d_model] + final_mlp_layers)
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, features, radiances, lon, lat, time, year):
        x = radiances.unfold(1, self.patch_size, self.stride)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.radiance_norm(self.radiance_linear(x))
        batch_size = x.size(0)
        feature_emb = self.feature_norm(self.uniform_linear(features))
        spatio_temporal_emb = self.spatio_temporal_encoder(lon, lat, time, year)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) * (feature_emb + spatio_temporal_emb).unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.seq_positional_encoder(x)
        x = self.transformer_encoder(x)
        return self.final_nn(x[:, 0, :])


class XCO2Dataset(Dataset):
    def __init__(self, X_radiance, X_features, y_diff, xco2_prior, lon, lat, time_days, years):
        self.radiance = torch.tensor(X_radiance, dtype=torch.float32)
        self.features = torch.tensor(X_features, dtype=torch.float32)
        self.labels = torch.tensor(y_diff, dtype=torch.float32)
        self.xco2_prior = torch.tensor(xco2_prior, dtype=torch.float32)
        self.lon = torch.tensor(lon, dtype=torch.float32)
        self.lat = torch.tensor(lat, dtype=torch.float32)
        self.time = torch.tensor(time_days, dtype=torch.float32)
        self.year = torch.tensor(years, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.features[idx], self.radiance[idx], self.labels[idx], self.xco2_prior[idx],
                self.lon[idx], self.lat[idx], self.time[idx], self.year[idx])


def _safe_value_for_csv(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _describe_bad_radiance(sample_values, band_name):
    nonfinite_mask = ~np.isfinite(sample_values)
    if not nonfinite_mask.any():
        return None
    first_bad_index = int(np.flatnonzero(nonfinite_mask)[0]) + 1
    return (
        f"{band_name}:nonfinite_count={int(nonfinite_mask.sum())},"
        f"first_bad_channel={first_bad_index}"
    )


def load_nc_file(path, feature_cols):
    invalid_records = []
    summary = {
        "source_file": path,
        "file_name": os.path.basename(path),
        "status": "success",
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "invalid_time": 0,
        "invalid_lon": 0,
        "invalid_lat": 0,
        "invalid_xco2": 0,
        "invalid_xco2_prior": 0,
        "invalid_feature": 0,
        "invalid_radiance_o2": 0,
        "invalid_radiance_weak": 0,
        "invalid_radiance_strong": 0,
        "error_message": ""
    }
    try:
        with xr.open_dataset(path) as ds:
            lon = ds["lon_L2"].values.astype(np.float32)
            lat = ds["lat_L2"].values.astype(np.float32)
            if "time_utc_L2" not in ds:
                raise ValueError("time_utc_L2 not found in nc file")
            raw_times = ds["time_utc_L2"].values
            try:
                times = pd.to_datetime(raw_times, format="mixed", errors="coerce")
            except (TypeError, ValueError):
                times = pd.to_datetime(raw_times, errors="coerce")
            time_invalid_mask = np.asarray(pd.isna(times), dtype=bool)
            time_days = np.asarray(times.dayofyear, dtype=np.float64)
            years = np.asarray(times.year, dtype=np.float64)

            radiance_o2 = np.stack(
                [ds[f"radiance_o2_{i + 1}"].values for i in range(1016)],
                axis=1
            ).astype(np.float32)
            radiance_weak = np.stack(
                [ds[f"radiance_weak_co2_{i + 1}"].values for i in range(1016)],
                axis=1
            ).astype(np.float32)
            radiance_strong = np.stack(
                [ds[f"radiance_strong_co2_{i + 1}"].values for i in range(1016)],
                axis=1
            ).astype(np.float32)

            X_features = np.stack(
                [ds[var].values for var in feature_cols],
                axis=1
            ).astype(np.float32)
            y_true = ds["xco2"].values.astype(np.float32)
            xco2_prior = ds["xco2_prior"].values.astype(np.float32)

            if "sounding_id" in ds:
                sounding_ids = np.asarray(ds["sounding_id"].values)
            else:
                sounding_ids = np.full(len(y_true), "", dtype=object)

        n_samples = len(y_true)
        summary["total_samples"] = int(n_samples)

        sample_lengths = {
            "lon": len(lon),
            "lat": len(lat),
            "time": len(times),
            "radiance_o2": len(radiance_o2),
            "radiance_weak": len(radiance_weak),
            "radiance_strong": len(radiance_strong),
            "features": len(X_features),
            "xco2": len(y_true),
            "xco2_prior": len(xco2_prior)
        }
        if len(set(sample_lengths.values())) != 1:
            raise ValueError(f"sample dimension mismatch: {sample_lengths}")

        lon_invalid = ~np.isfinite(lon)
        lat_invalid = ~np.isfinite(lat)
        xco2_invalid = ~np.isfinite(y_true)
        prior_invalid = ~np.isfinite(xco2_prior)
        feature_bad_matrix = ~np.isfinite(X_features)
        feature_invalid = feature_bad_matrix.any(axis=1)

        o2_bad_matrix = ~np.isfinite(radiance_o2)
        weak_bad_matrix = ~np.isfinite(radiance_weak)
        strong_bad_matrix = ~np.isfinite(radiance_strong)
        o2_invalid = o2_bad_matrix.any(axis=1)
        weak_invalid = weak_bad_matrix.any(axis=1)
        strong_invalid = strong_bad_matrix.any(axis=1)

        invalid_mask = (
            time_invalid_mask
            | lon_invalid
            | lat_invalid
            | xco2_invalid
            | prior_invalid
            | feature_invalid
            | o2_invalid
            | weak_invalid
            | strong_invalid
        )
        valid_mask = ~invalid_mask

        summary.update({
            "valid_samples": int(valid_mask.sum()),
            "invalid_samples": int(invalid_mask.sum()),
            "invalid_time": int(time_invalid_mask.sum()),
            "invalid_lon": int(lon_invalid.sum()),
            "invalid_lat": int(lat_invalid.sum()),
            "invalid_xco2": int(xco2_invalid.sum()),
            "invalid_xco2_prior": int(prior_invalid.sum()),
            "invalid_feature": int(feature_invalid.sum()),
            "invalid_radiance_o2": int(o2_invalid.sum()),
            "invalid_radiance_weak": int(weak_invalid.sum()),
            "invalid_radiance_strong": int(strong_invalid.sum())
        })

        invalid_indices = np.flatnonzero(invalid_mask)
        for idx in invalid_indices:
            reasons = []
            invalid_variables = []
            if time_invalid_mask[idx]:
                reasons.append("time_parse_failed")
                invalid_variables.append("time_utc_L2")
            if lon_invalid[idx]:
                reasons.append("nonfinite_lon")
                invalid_variables.append("lon_L2")
            if lat_invalid[idx]:
                reasons.append("nonfinite_lat")
                invalid_variables.append("lat_L2")
            if xco2_invalid[idx]:
                reasons.append("nonfinite_xco2")
                invalid_variables.append("xco2")
            if prior_invalid[idx]:
                reasons.append("nonfinite_xco2_prior")
                invalid_variables.append("xco2_prior")
            if feature_invalid[idx]:
                bad_feature_indices = np.flatnonzero(feature_bad_matrix[idx])
                for feature_idx in bad_feature_indices:
                    feature_name = feature_cols[int(feature_idx)]
                    reasons.append(f"feature_nonfinite:{feature_name}")
                    invalid_variables.append(feature_name)
            for band_values, band_name in [
                (radiance_o2[idx], "radiance_o2"),
                (radiance_weak[idx], "radiance_weak_co2"),
                (radiance_strong[idx], "radiance_strong_co2")
            ]:
                description = _describe_bad_radiance(band_values, band_name)
                if description is not None:
                    reasons.append(description)
                    invalid_variables.append(band_name)
            raw_time_value = raw_times[idx]
            invalid_records.append({
                "source_file": path,
                "file_name": os.path.basename(path),
                "sample_index_in_file": int(idx),
                "sounding_id": _safe_value_for_csv(sounding_ids[idx]),
                "raw_time": str(raw_time_value),
                "lon": _safe_value_for_csv(lon[idx]),
                "lat": _safe_value_for_csv(lat[idx]),
                "xco2": _safe_value_for_csv(y_true[idx]),
                "xco2_prior": _safe_value_for_csv(xco2_prior[idx]),
                "invalid_variables": ";".join(dict.fromkeys(invalid_variables)),
                "invalid_reasons": ";".join(reasons)
            })

        if summary["invalid_samples"] > 0:
            print(f"[WARNING] {os.path.basename(path)}: discard {summary['invalid_samples']}/{summary['total_samples']} invalid samples")
        if valid_mask.sum() == 0:
            summary["status"] = "no_valid_samples"
            gc.collect()
            return None, invalid_records, summary

        lon = lon[valid_mask]
        lat = lat[valid_mask]
        time_days = time_days[valid_mask].astype(np.float32)
        years = years[valid_mask].astype(np.float32)
        radiance_o2 = radiance_o2[valid_mask]
        radiance_weak = radiance_weak[valid_mask]
        radiance_strong = radiance_strong[valid_mask]
        X_features = X_features[valid_mask]
        y_true = y_true[valid_mask]
        xco2_prior = xco2_prior[valid_mask]

        X_radiance = np.concatenate(
            [radiance_o2, radiance_weak, radiance_strong],
            axis=1
        ).astype(np.float32)
        y_diff = y_true - xco2_prior

        lon_norm, lat_norm, time_norm, year_norm = normalize_spatio_temporal_data(
            lon, lat, time_days, years, reference_year=2017
        )
        gc.collect()
        data = (
            X_radiance,
            X_features,
            y_diff,
            xco2_prior,
            lon_norm,
            lat_norm,
            time_norm,
            year_norm
        )
        return data, invalid_records, summary
    except Exception as e:
        summary["status"] = "read_failed"
        summary["error_message"] = str(e)
        print(f"[WARNING] Failed reading {path}: {e}")
        gc.collect()
        return None, invalid_records, summary


def evaluate_and_visualize(model, dataloader, device, save_dir, save_prefix):
    model.eval()
    y_true_diff, y_pred_diff = [], []
    xco2_prior_list = []
    with torch.no_grad():
        for features, radiance, label_diff, prior, lon, lat, time, year in tqdm(dataloader, desc="Predicting"):
            features, radiance = features.to(device), radiance.to(device)
            lon, lat, time, year = lon.to(device), lat.to(device), time.to(device), year.to(device)
            pred_diff = model(features, radiance, lon, lat, time, year).squeeze(-1).float().cpu().numpy()
            label_diff = label_diff.numpy().reshape(-1)
            prior = prior.numpy().reshape(-1)
            pred_diff = pred_diff.reshape(-1)
            y_true_diff.append(label_diff)
            y_pred_diff.append(pred_diff)
            xco2_prior_list.append(prior)

    y_true_diff = np.concatenate(y_true_diff)
    y_pred_diff = np.concatenate(y_pred_diff)
    xco2_prior = np.concatenate(xco2_prior_list)
    y_true_xco2 = y_true_diff + xco2_prior
    y_pred_xco2 = y_pred_diff + xco2_prior

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{save_prefix}_predictions.csv")
    pd.DataFrame({
        "xco2_prior": xco2_prior.round(4),
        "true_diff": y_true_diff.round(4),
        "pred_diff": y_pred_diff.round(4),
        "true_xco2": y_true_xco2.round(4),
        "pred_xco2": y_pred_xco2.round(4)
    }).to_csv(csv_path, index=False)
    print(f"[INFO] Prediction saved to: {csv_path}")

    def plot_scatter(x, y, xlabel, ylabel, save_name):
        R2 = r2_score(x, y)
        MAE = mean_absolute_error(x, y)
        RMSE = np.sqrt(mean_squared_error(x, y))
        slope, intercept, r_val, _, _ = linregress(x, y)
        plt.figure(figsize=(8, 6))
        plt.hexbin(x, y, gridsize=200, cmap='jet', mincnt=1)
        plt.colorbar(label='density')
        plt.plot(x, x, 'r--', label='1:1 line')
        plt.plot(x, slope * x + intercept, 'b--', label=f'fit: y={slope:.4f}x+{intercept:.4f}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.text(0.95, 0.95, f"R: {r_val:.3f}\nMAE: {MAE:.4f}\nRMSE: {RMSE:.4f}\nN={len(x)}",
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.6),
                 ha='right', va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close()

    plot_scatter(y_true_diff, y_pred_diff,
                 "True difference (XCO2 - prior, ppm)",
                 "Predicted difference (XCO2 - prior, ppm)",
                 f"{save_prefix}_diff_scatter.png")
    plot_scatter(y_true_xco2, y_pred_xco2,
                 "True XCO2 (ppm)",
                 "Predicted XCO2 (ppm)",
                 f"{save_prefix}_xco2_scatter.png")


def main():
    if sys.platform == "win32":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_root", type=str, required=True, help="Root directory containing netCDF files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for model and results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    root_dir = args.nc_root
    save_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    os.makedirs(save_dir, exist_ok=True)

    feature_cols = [
        "xco2_prior",
        "azimuth", "zenith", "solar_azimuth", "solar_zenith",
        "polar_angle", "solar_distance", "relative_velocity"
    ]

    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".nc"):
                files.append(os.path.join(root, filename))

    all_radiance, all_features, all_diff, all_prior = [], [], [], []
    all_lon, all_lat, all_time, all_year = [], [], [], []
    all_invalid_records = []
    all_file_summaries = []

    for f in tqdm(files, desc=f"Loading {len(files)} nc files"):
        data, invalid_records, file_summary = load_nc_file(f, feature_cols)
        all_invalid_records.extend(invalid_records)
        all_file_summaries.append(file_summary)
        if data is not None:
            r, feat, diff, prior, lon, lat, time, year = data
            all_radiance.append(r)
            all_features.append(feat)
            all_diff.append(diff)
            all_prior.append(prior)
            all_lon.append(lon)
            all_lat.append(lat)
            all_time.append(time)
            all_year.append(year)

    invalid_columns = [
        "source_file", "file_name", "sample_index_in_file", "sounding_id",
        "raw_time", "lon", "lat", "xco2", "xco2_prior",
        "invalid_variables", "invalid_reasons"
    ]
    invalid_df = pd.DataFrame(all_invalid_records, columns=invalid_columns)
    invalid_detail_path = os.path.join(save_dir, "invalid_samples_detail.csv")
    invalid_df.to_csv(invalid_detail_path, index=False, encoding="utf-8-sig")

    file_summary_df = pd.DataFrame(all_file_summaries)
    file_summary_path = os.path.join(save_dir, "invalid_summary_by_file.csv")
    file_summary_df.to_csv(file_summary_path, index=False, encoding="utf-8-sig")

    from collections import Counter
    reason_counter = Counter()
    for record in all_invalid_records:
        for reason in record["invalid_reasons"].split(";"):
            if reason:
                reason_key = reason.split(":", 1)[0]
                reason_counter[reason_key] += 1
    reason_summary_df = pd.DataFrame(
        [
            {"invalid_reason": reason, "affected_sample_count": count}
            for reason, count in reason_counter.most_common()
        ],
        columns=["invalid_reason", "affected_sample_count"]
    )
    reason_summary_path = os.path.join(save_dir, "invalid_summary_by_reason.csv")
    reason_summary_df.to_csv(reason_summary_path, index=False, encoding="utf-8-sig")

    total_raw_samples = int(file_summary_df["total_samples"].sum()) if not file_summary_df.empty else 0
    total_valid_samples = int(file_summary_df["valid_samples"].sum()) if not file_summary_df.empty else 0
    total_invalid_samples = int(file_summary_df["invalid_samples"].sum()) if not file_summary_df.empty else 0
    failed_file_count = int((file_summary_df["status"] == "read_failed").sum()) if not file_summary_df.empty else 0

    print("\n" + "=" * 80)
    print("Data filter statistics")
    print(f"Total raw samples: {total_raw_samples}")
    print(f"Valid samples: {total_valid_samples}")
    print(f"Invalid samples: {total_invalid_samples}")
    if total_raw_samples > 0:
        print(f"Invalid ratio: {total_invalid_samples / total_raw_samples * 100:.6f}%")
    print(f"Failed file count: {failed_file_count}")
    print(f"Invalid detail: {invalid_detail_path}")
    print(f"Summary per file: {file_summary_path}")
    print(f"Summary per reason: {reason_summary_path}")
    print("=" * 80 + "\n")

    if not all_radiance:
        raise ValueError("No valid data loaded after filtering, check invalid output csv")

    X_radiance = np.concatenate(all_radiance, axis=0)
    X_features = np.concatenate(all_features, axis=0)
    y_diff = np.concatenate(all_diff, axis=0)
    xco2_prior = np.concatenate(all_prior, axis=0)
    lon = np.concatenate(all_lon, axis=0)
    lat = np.concatenate(all_lat, axis=0)
    time = np.concatenate(all_time, axis=0)
    year = np.concatenate(all_year, axis=0)

    print("[INFO] Calculating normalization statistics in float64")
    radiance_mean = np.mean(X_radiance, axis=0, dtype=np.float64)
    radiance_std = np.std(X_radiance, axis=0, dtype=np.float64)
    feature_mean = np.mean(X_features, axis=0, dtype=np.float64)
    feature_std = np.std(X_features, axis=0, dtype=np.float64)

    if not np.all(np.isfinite(radiance_mean)):
        raise ValueError("radiance_mean contains NaN or Inf")
    if not np.all(np.isfinite(radiance_std)):
        raise ValueError("radiance_std contains NaN or Inf")
    if not np.all(np.isfinite(feature_mean)):
        raise ValueError("feature_mean contains NaN or Inf")
    if not np.all(np.isfinite(feature_std)):
        raise ValueError("feature_std contains NaN or Inf")

    radiance_std = np.maximum(radiance_std, 1e-6)
    feature_std = np.maximum(feature_std, 1e-6)

    radiance_mean = radiance_mean.astype(np.float32)
    radiance_std = radiance_std.astype(np.float32)
    feature_mean = feature_mean.astype(np.float32)
    feature_std = feature_std.astype(np.float32)

    normalization_stats_path = os.path.join(save_dir, "normalization_stats.npz")
    np.savez(
        normalization_stats_path,
        radiance_mean=radiance_mean,
        radiance_std=radiance_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_cols=np.array(feature_cols),
        reference_year=np.array(2017, dtype=np.int32)
    )
    print(f"[INFO] Normalization stats saved: {normalization_stats_path}")
    print(
        f"Radiance stat range: mean=[{radiance_mean.min():.6e}, {radiance_mean.max():.6e}], "
        f"std=[{radiance_std.min():.6e}, {radiance_std.max():.6e}]"
    )
    print(
        f"Feature stat range: mean=[{feature_mean.min():.6e}, {feature_mean.max():.6e}], "
        f"std=[{feature_std.min():.6e}, {feature_std.max():.6e}]"
    )

    X_radiance = (X_radiance - radiance_mean) / radiance_std
    X_features = (X_features - feature_mean) / feature_std

    split_data = train_test_split(
        X_radiance, X_features, y_diff, xco2_prior, lon, lat, time, year,
        test_size=0.2, random_state=args.seed
    )
    Xr_tr, Xr_val, Xf_tr, Xf_val, y_tr, y_val, prior_tr, prior_val, \
    lon_tr, lon_val, lat_tr, lat_val, time_tr, time_val, year_tr, year_val = split_data

    train_ds = XCO2Dataset(Xr_tr, Xf_tr, y_tr, prior_tr, lon_tr, lat_tr, time_tr, year_tr)
    val_ds = XCO2Dataset(Xr_val, Xf_val, y_val, prior_val, lon_val, lat_val, time_val, year_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(
        patch_size=16,
        stride=8,
        feature_dim=len(feature_cols),
        d_model=128,
        nhead=8,
        num_transformer_layers=8,
        dim_feedforward=128,
        final_mlp_layers=[64, 1],
        input_radiance_length=3048,
        spatial_freq=16,
        temporal_freq=16
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_enabled = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float('inf')
    save_model_path = os.path.join(save_dir, "best_model_with_st_encoding.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, radiance, label, _, lon, lat, time, year in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features, radiance, label = features.to(device), radiance.to(device), label.to(device)
            lon, lat, time, year = lon.to(device), lat.to(device), time.to(device), year.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled
            ):
                pred = model(features, radiance, lon, lat, time, year).squeeze()
                loss = criterion(pred, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * features.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, radiance, label, _, lon, lat, time, year in val_loader:
                features, radiance, label = features.to(device), radiance.to(device), label.to(device)
                lon, lat, time, year = lon.to(device), lat.to(device), time.to(device), year.to(device)
                pred = model(features, radiance, lon, lat, time, year).squeeze()
                val_loss += criterion(pred, label).item() * features.size(0)

        avg_train_loss = train_loss / len(train_ds)
        avg_val_loss = val_loss / len(val_ds)
        print(f"Epoch {epoch + 1} | Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_path)
            print(f"[SAVE] Best model saved, val loss: {best_val_loss:.6f}")
        scheduler.step()

    model.load_state_dict(torch.load(save_model_path, map_location=device))
    evaluate_and_visualize(model, val_loader, device, save_dir, "xco2_results_with_st_encoding")


if __name__ == "__main__":
    main()
