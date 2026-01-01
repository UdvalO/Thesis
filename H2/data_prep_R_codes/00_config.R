# ==============================================================================
# 00_config.R (H2 Proposed Model)
# ==============================================================================

# --- 1. PROJECT ROOT ---
# Define root path
PROJECT_ROOT <- "..."

BASE_DIR     <- ".../dataset"   # to load from
SAVE_DIR     <- ".../data_processed" # to save into

# Make sure the save directory exists
if (!dir.exists(SAVE_DIR)) {
  cat(sprintf("Creating save directory: %s\n", SAVE_DIR))
  dir.create(SAVE_DIR, recursive = TRUE)
}

# Vintages years in according to the paper
VINTAGE_YEARS <- 2009:2010 

# --- 3. SAVE FILE NAMES ---
ORIG_SAVE_FILE <- file.path(SAVE_DIR, "orig_data_cleaned.rds")
PERF_SAVE_FILE <- file.path(SAVE_DIR, "perf_data_cleaned.rds")

# --- 4. TIME WINDOWS ---
# Training: Jan 2012 - June 2013 (18 months) 
# Testing: July 2013 - Dec 2013 (6 months
PERF_END_PERIOD <- 201412       
PERF_START_HISTORY <- 201107  # Need 6 months buffer before Jan 2012 ("Burn-in")
START_PERIOD_TRAIN <- 201201    
TEST_END_DATE <- 201312         
TRAIN_SPLIT_DATE <- 201306      

# --- 5. DATA CLEANING CONSTANTS ---
NA_STRINGS <- c("", "999", "9999", "99", "9")

# --- 6. COLUMN MAPS ---
orig_cols_map <- c(
  Loan_Sequence_Number = 20,  # Loan identifier
  fico = 1,              # Credit score
  First_Pmt_Date = 2,    # First payment date
  if_fthb = 3,           # First time home buyer
  mi_pct = 6,            # Mortgage insurance %
  cnt_units = 7,         # Number of units
  if_prim_res = 8,       # Occupancy (Primary)
  dti = 10,              # Debt-to-income
  ltv = 12,              # Loan-to-value                  
  if_corr = 14,          # Channel (Correspondent)      
  Postal_Code = 19,      # Geo Key Source     
  if_sf = 18,            # Property type (Single Family)     
  if_purc = 21,          # Loan purpose (Purchase)
  cnt_borr = 23,         # Num borrowers
  Seller_Name = 24,      # Lender Key Source
  if_sc = 26,            # Super conforming (Table 1: if_se)          
  Servicer_Name = 25,    # Current management entity      
  orig_upb = 11          # Original unpaid balance   
)

# [CHANGE]: Renamed col 8 to Modification_Flag and added the missing comma
perf_cols_map <- c(
  Loan_Sequence_Number = 1,     # Loan identifier
  Monthly_Reporting_Period = 2, # Date of records
  current_upb = 3,              # Current unpaid balance
  if_delq_sts = 4,              # Delinquency status
  Loan_Age = 5,                 # Loan Age, resets upon modification
  mths_remng = 6,               # Months remaining
  Modification_Flag = 8,        # Flag if loan has been modified
  current_int_rt = 11           # Current interest rate
)

# --- 7. COLUMN TYPE LISTS ---
ORIG_NUMERIC_COLS <- c("fico", "mi_pct", "cnt_units", "dti", "ltv", 
                       "cnt_borr", "First_Pmt_Date", "orig_upb")

PERF_NUMERIC_COLS <- c("current_upb", "mths_remng", "current_int_rt", "Loan_Age")

# Combined
ALL_NUMERIC_COLS_FINAL <- c(
  "fico", "mi_pct", "cnt_units", "dti", "ltv", 
  "cnt_borr", "First_Pmt_Date", "orig_upb",
  "current_upb", "mths_remng", "current_int_rt", "Loan_Age"
)

# Final Scaling Lists
FINAL_ORIG_COLS_TO_SCALE <- c("fico", "mi_pct", "cnt_units", "dti", "ltv", "cnt_borr")
FINAL_PERF_COLS_TO_SCALE <- c("upb_pct_remaining", "mths_remng", 
                              "current_int_rt", "clean_loan_age")

FINAL_SCALE_COLS <- c(FINAL_ORIG_COLS_TO_SCALE, FINAL_PERF_COLS_TO_SCALE)