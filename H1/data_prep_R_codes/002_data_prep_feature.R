# ==============================================================================
# 002_data_prep_feature.R (H1)
#
# PURPOSE:
#   Prepare model-ready features from cleaned origination and performance data.
#
# STEPS:
#   1. Load cleaned origination and performance RDS files.
#   2. Sample loans to match target scale and default rate (250k, 5% default).
#   3. Engineer stratification keys (geography and lender).
#   4. Apply leakage-safe capping and median imputation.
#   5. Encode binary indicator variables.
#   6. Derive default timing features for downstream graph logic.
#   7. Merge origination and performance features.
#   8. Save final feature dataset as RDS.
#
# INPUTS: 
#   - 000_config.R
#   - orig_data_cleaned.rds (ORIG_SAVE_FILE)
#   - perf_data_cleaned.rds (PERF_SAVE_FILE)
#
# OUTPUTS: 
#   - final_features_h1.rds (FINAL_FEATURES_FILE)
# ==============================================================================

# --- 0. DEPENDENCIES ---
source("000_config.R")
library(data.table)
library(tidyverse)
set.seed(42) # for reproducibility

# --- HELPER FUNCTIONS -------------------------------------------------------
# Performs leakage-safe outlier capping and median imputation
perform_capping_and_imputation <- function(DT, numeric_cols, train_filter_expr = NULL) {
  # Paper: Cap outliers at 1st and 99th percentile [cite: 368]
  # Paper: Median imputation for nulls [cite: 369]
  
  for (col in numeric_cols) {
    if (!col %in% names(DT)) next
    
    # 1. Define the population for statistics (TRAIN ONLY)
    if (!is.null(train_filter_expr)) {
      stats_data <- DT[eval(train_filter_expr)][[col]]
    } else {
      stats_data <- DT[[col]]
    }
    
    # Compute capping thresholds and median
    q <- quantile(stats_data, probs = c(0.01, 0.99), na.rm = TRUE)
    med_val <- median(stats_data, na.rm = TRUE)
    
    # Apply transformations to full dataset
    DT[get(col) < q[1], (col) := q[1]]
    DT[get(col) > q[2], (col) := q[2]]
    DT[is.na(get(col)), (col) := med_val]
  }
  return(DT)
}

# --- 1. LOAD CLEANED DATA ---------------------------------------------------
cat("--- Loading Data ---\n")

# Files created in 001 script
orig_data <- readRDS(ORIG_SAVE_FILE)
perf_data <- readRDS(PERF_SAVE_FILE)
setDT(orig_data); setDT(perf_data)

# --- 2. SAMPLING STRATEGY ---------------------------------------------------
# Goal: Replicate paper scale
# - 250k loans
# - 5% default rate (ever 90+ days delinquent)

cat("--- Performing Sampling ---\n")

# Define Defaulters: Ever 90+ days delinquent
defaulter_ids <- perf_data[if_delq_sts %in% c("3", "4", "5", "6", "7", "8", "9", "R", "RA"), 
                           unique(Loan_Sequence_Number)]

# Stratification Keys for Non-Defaulters
orig_data[, Geo_Key := substr(as.character(Postal_Code), 1, 2)]
orig_data[, Lender_Key := Seller_Name]

# Define positive and negative pools
pos_pool <- intersect(defaulter_ids, orig_data$Loan_Sequence_Number)
neg_pool <- setdiff(orig_data$Loan_Sequence_Number, pos_pool)

# Target sample sizes
N_TOTAL <- 250000
DEFAULT_RATE <- 0.05   # 5% as observed in the Reference Model

N_POS <- round(N_TOTAL * DEFAULT_RATE)
N_NEG <- N_TOTAL - N_POS

# Sample defaulters
sampled_pos <- sample(pos_pool, min(length(pos_pool), N_POS), replace = FALSE)

# Stratified sampling for non-defaulters (Geo C Lender)
neg_data <- orig_data[Loan_Sequence_Number %in% neg_pool, .(Loan_Sequence_Number, Geo_Key, Lender_Key)]
neg_data[, strata := paste(Geo_Key, Lender_Key, sep="_")]
strata_counts <- neg_data[, .N, by=strata]
strata_counts[, target_n := round(N / sum(N) * N_NEG)]

sampled_neg <- neg_data[, .SD[sample(.N, min(.N, strata_counts[strata == .BY$strata, target_n]))], 
                        by=strata]$Loan_Sequence_Number

# Top-up if stratified sample falls short
if(length(sampled_neg) < N_NEG) {
  rem <- setdiff(neg_pool, sampled_neg)
  sampled_neg <- c(sampled_neg, sample(rem, N_NEG - length(sampled_neg)))
}

final_ids <- c(sampled_pos, sampled_neg)
cat(sprintf("Final Sample: %d loans\n", length(final_ids)))

# Filter datasets to sampled loans
orig_data <- orig_data[Loan_Sequence_Number %in% final_ids]
perf_data <- perf_data[Loan_Sequence_Number %in% final_ids]

# --- 3. FEATURE ENGINEERING  -----------------------------------------------
cat("--- Engineering Features ---\n")

TRAIN_CUTOFF <- 201306

# A. Leakage-safe capping & imputation
orig_data <- perform_capping_and_imputation(orig_data, ORIG_NUMERIC_COLS)

# Imputing based only on training to prevent leakage
perf_expr <- quote(Monthly_Reporting_Period <= TRAIN_CUTOFF)
perf_data <- perform_capping_and_imputation(perf_data, PERF_NUMERIC_COLS, perf_expr)

# B. Binary indicators (coalesce missing to 0)
bin_cols <- c("if_fthb", "if_prim_res", "if_corr", "if_sf", "if_purc", "if_sc")
orig_data[, (bin_cols) := lapply(.SD, function(x) ifelse(is.na(x), 0, x)), .SDcols = bin_cols]

# C. Default month calculation (for downstream graph logic)
perf_data[, delq_numeric := fcase(
  if_delq_sts %in% c("R", "RA"), 3L,
  if_delq_sts %in% c("1", "2"), as.integer(if_delq_sts),
  if_delq_sts == "0", 0L,
  default = 3L 
)]

def_events <- perf_data[delq_numeric >= 3, 
                        .(Default_Month = min(Monthly_Reporting_Period)), 
                        by = Loan_Sequence_Number]

# Attach default month (999999 = never defaulted)
perf_data <- merge(perf_data, def_events, by="Loan_Sequence_Number", all.x=TRUE)
perf_data[is.na(Default_Month), Default_Month := 999999] # placeholder to avoid using NA

# --- 4. MERGE FEATURES & SAVE ----------------------------------------------
cat("--- Merging & Saving ---\n")

# Keep graph-relevant attributes
cols_keep_orig <- c("Loan_Sequence_Number", ORIG_NUMERIC_COLS, bin_cols, "Geo_Key", "Lender_Key")

final_dt <- merge(perf_data, orig_data[, ..cols_keep_orig], by="Loan_Sequence_Number", all.x=TRUE)

saveRDS(final_dt, FINAL_FEATURES_FILE)
cat("Saved to:", FINAL_FEATURES_FILE, "\n")