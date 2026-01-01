# ==============================================================================
# 003_prep_split_scale.R (H1)
#
# PURPOSE:
#   Define strict train/test time windows, derive prediction targets, and compute
#   scaling parameters for H1 modeling.
#
# STEPS:
#   1. Load final engineered feature dataset.
#   2. Define explicit training and testing snapshot windows.
#   3. Derive binary default targets using a fixed 12-month prediction horizon.
#   4. Mask loans already in default at snapshot time.
#   5. Compute scaling parameters using training data only (leakage-safe).
#   6. Save targets, preprocessing parameters, and final feature data.
#
# INPUTS:
#   - final_features_h1.rds
#   - 000_config.R
#
# OUTPUTS:
#   - train_targets.rds
#   - test_targets.rds
#   - preproc_params.rds
#   - final_features_h1.rds
# ==============================================================================

# --- 0. DEPENDENCIES ---
source("000_config.R")
library(data.table)
library(lubridate)

# --- LOAD FINAL FEATURE DATA -----------------------------------------------
# Input produced by 002_data_prep_feature.R
cat("--- Loading Final H1 Features ---\n")
final_data <- readRDS(FINAL_FEATURES_FILE)
setDT(final_data)

# --- 1. DEFINE STRICT TRAIN / TEST TIMELINES --------------------------------
# H1 requires explicit, non-overlapping time windows
# Training: Jan 2012 b