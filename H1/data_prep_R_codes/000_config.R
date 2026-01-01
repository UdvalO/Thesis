# ==============================================================================
# 000_config.R (H1 CONSTRAINT-AWARE REPLICATION)
# ==============================================================================

# --- 1. PROJECT ROOT ---
# Define root path
PROJECT_ROOT <- "...\\H1"

BASE_DIR     <- "...H1/dataset"   # to load from
SAVE_DIR     <- "...H1/data_processed" # to save into

# Make sure the save directory exists
if(!dir.exists(SAVE_DIR)) dir.create(SAVE_DIR, recursive = TRUE)

# Vintages years in according to the paper
VINTAGE_YEARS <- 2009:2010 

# --- 3. SAVE FILE NAMES ---
ORIG_SAVE_FILE <- file.path(SAVE_DIR, "orig_data_cleaned.rds")
PERF_SAVE_FILE <- file.path(SAVE_DIR, "perf_data_cleaned.rds")
FINAL_FEATURES_FILE <- file.path(SAVE_DIR, "final_features_h1.rds")

# Target & Param Files (for validation script to find)
TRAIN_TARGETS_FILE  <- file.path(SAVE_DIR, "train_targets.rds")
TEST_TARGETS_FILE   <- file.path(SAVE_DIR, "test_targets.rds")
PREPROC_PARAMS_FILE <- file.path(SAVE_DIR, "preproc_params.rds")

# --- 4. TIME WINDOWS ---
# Training: Jan 2012 - June 2013 (18 months) 
# Testing: July 2013 - Dec 2013 (6 months)
PERF_START_HISTORY <- 201107   # Need 6 months buffer before Jan 2012 ("Burn-in")
START_PERIOD_TRAIN <- 201201   
TRAIN_SPLIT_DATE   <- 201306   
TEST_END_DATE      <- 201312   
PERF_END_PERIOD    <- 201412   # Horizon for labels

# --- 5. CLEANING CONSTANTS ---
NA_STRINGS <- c("", "999", "9999", "99", "9")

# --- 6. COLUMN MAPS ---
# Strictly aligning with Table 1 in Reference Model
orig_cols_map <- c(
  Loan_Sequence_Number = 20, # Loan identifier
  fico = 1,              # Credit score
  First_Pmt_Date = 2,    # First payment date
  if_fthb = 3,           # First time home buyer
  mi_pct = 6,            # Mortgage insurance %
  cnt_units = 7,         # Number of units
  if_prim_res = 8,       # Occupancy (Primary)
  dti = 10,              # Debt-to-income
  ltv = 12,              # Loan-to-value
  if_corr = 14,          # Channel (Correspondent)
  if_sf = 18,            # Property type (Single Family)
  Postal_Code = 19,      # Geo Key Source
  if_purc = 21,          # Loan purpose (Purchase)
  cnt_borr = 23,         # Num borrowers
  Seller_Name = 24,      # Lender Key Source
  if_sc = 26             # Super conforming (Table 1: if_se)
)

perf_cols_map <- c(
  Loan_Sequence_Number = 1,
  Monthly_Reporting_Period = 2,
  current_upb = 3,       # Current unpaid balance
  if_delq_sts = 4,       # Delinquency status
  mths_remng = 6,        # Months remaining
  current_int_rt = 11    # Current interest rate
)

# --- 7. FEATURE LISTS (STRICT H1) ---

# Numeric columns to cap/impute in Origination
ORIG_NUMERIC_COLS <- c("fico", "mi_pct", "cnt_units", "dti", "ltv", "cnt_borr")

# Numeric columns to cap/impute in Performance
PERF_NUMERIC_COLS <- c("current_upb", "mths_remng", "current_int_rt")

# Final features to be scaled (0-1) for the GNN
# Matches Table 1 non-binary features in the Reference Model
FINAL_SCALE_COLS <- c(
  "fico", "mi_pct", "cnt_units", "dti", "ltv", "cnt_borr",
  "current_upb", "mths_remng", "current_int_rt"
)
