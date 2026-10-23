# Canonical IPL Venue & City Mapping

This document details the standardization applied to stadium names and missing city imputation across all 1,243 matches in the Cricsheet IPL dataset.

---

## 1. Standardization Methodology & Rules

1. **Physical Ground Consolidation:**
   - Multiple naming variations of the exact same physical cricket ground (e.g. variations with city suffixes like `, Kolkata`, `, Mumbai`, or punctuation like `M.Chinnaswamy` vs `M Chinnaswamy`) are consolidated into one canonical venue name.
   - Ground renamings where the physical facility is identical (e.g. `Feroz Shah Kotla` $\rightarrow$ `Arun Jaitley Stadium`, `Subrata Roy Sahara Stadium` $\rightarrow$ `Maharashtra Cricket Association Stadium`, `Sardar Patel Stadium, Motera` $\rightarrow$ `Narendra Modi Stadium`) are mapped to the modern canonical name.
2. **Preservation of Distinct Grounds:**
   - Grounds in the same city that are physically distinct (e.g. Mumbai's `Wankhede Stadium`, `Brabourne Stadium`, and `Dr DY Patil Sports Academy`) remain strictly distinct venues.
3. **Conservative City Imputation:**
   - Missing `city` attributes are only imputed when the venue provides an unambiguous, 100% certain geographic location:
     - `Dubai International Cricket Stadium` $\rightarrow$ `Dubai` (33 matches imputed)
     - `Sharjah Cricket Stadium` $\rightarrow$ `Sharjah` (18 matches imputed)
   - Ambiguous entries are never blindly guessed.

---

## 2. Complete Venue Mapping Table (60 Raw Strings $\rightarrow$ 36 Canonical Stadiums)

| Raw Venue Name in Dataset | Canonical Standardized Venue | Default / Canonical City | Notes |
| :--- | :--- | :--- | :--- |
| `M Chinnaswamy Stadium` | `M Chinnaswamy Stadium` | Bengaluru / Bangalore | Consolidated alias |
| `M Chinnaswamy Stadium, Bengaluru` | `M Chinnaswamy Stadium` | Bengaluru | Consolidated alias |
| `M.Chinnaswamy Stadium` | `M Chinnaswamy Stadium` | Bengaluru | Punctuation variant |
| `Eden Gardens` | `Eden Gardens` | Kolkata | Consolidated alias |
| `Eden Gardens, Kolkata` | `Eden Gardens` | Kolkata | City suffix removed |
| `Wankhede Stadium` | `Wankhede Stadium` | Mumbai | Consolidated alias |
| `Wankhede Stadium, Mumbai` | `Wankhede Stadium` | Mumbai | City suffix removed |
| `Brabourne Stadium` | `Brabourne Stadium` | Mumbai | Distinct Mumbai ground |
| `Brabourne Stadium, Mumbai` | `Brabourne Stadium` | Mumbai | City suffix removed |
| `Dr DY Patil Sports Academy` | `Dr DY Patil Sports Academy` | Mumbai / Navi Mumbai | Distinct Mumbai ground |
| `Dr DY Patil Sports Academy, Mumbai` | `Dr DY Patil Sports Academy` | Mumbai | City suffix removed |
| `Feroz Shah Kotla` | `Arun Jaitley Stadium` | Delhi | Renamed Sep 2019 (same ground) |
| `Arun Jaitley Stadium` | `Arun Jaitley Stadium` | Delhi | Canonical name |
| `Arun Jaitley Stadium, Delhi` | `Arun Jaitley Stadium` | Delhi | City suffix removed |
| `Rajiv Gandhi International Stadium` | `Rajiv Gandhi International Stadium` | Hyderabad | Canonical name |
| `Rajiv Gandhi International Stadium, Uppal` | `Rajiv Gandhi International Stadium` | Hyderabad | Locality suffix removed |
| `Rajiv Gandhi International Stadium, Uppal, Hyderabad` | `Rajiv Gandhi International Stadium` | Hyderabad | Locality/city suffix removed |
| `MA Chidambaram Stadium` | `MA Chidambaram Stadium` | Chennai | Canonical name |
| `MA Chidambaram Stadium, Chepauk` | `MA Chidambaram Stadium` | Chennai | Locality suffix removed |
| `MA Chidambaram Stadium, Chepauk, Chennai` | `MA Chidambaram Stadium` | Chennai | Locality/city suffix removed |
| `Sardar Patel Stadium, Motera` | `Narendra Modi Stadium` | Ahmedabad | Rebuilt & renamed 2021 |
| `Narendra Modi Stadium, Ahmedabad` | `Narendra Modi Stadium` | Ahmedabad | Canonical name |
| `Punjab Cricket Association Stadium, Mohali` | `Punjab Cricket Association IS Bindra Stadium` | Mohali / Chandigarh | Renamed after I.S. Bindra |
| `Punjab Cricket Association IS Bindra Stadium` | `Punjab Cricket Association IS Bindra Stadium` | Mohali | Canonical name |
| `Punjab Cricket Association IS Bindra Stadium, Mohali` | `Punjab Cricket Association IS Bindra Stadium` | Mohali | City suffix removed |
| `Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh` | `Punjab Cricket Association IS Bindra Stadium` | Mohali | Combined suffix removed |
| `Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur` | `Maharaja Yadavindra Singh International Cricket Stadium` | Mullanpur / New Chandigarh | Locality suffix removed |
| `Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh` | `Maharaja Yadavindra Singh International Cricket Stadium` | New Chandigarh | Locality suffix removed |
| `Maharashtra Cricket Association Stadium` | `Maharashtra Cricket Association Stadium` | Pune | Canonical name |
| `Maharashtra Cricket Association Stadium, Pune` | `Maharashtra Cricket Association Stadium` | Pune | City suffix removed |
| `Subrata Roy Sahara Stadium` | `Maharashtra Cricket Association Stadium` | Pune | Original 2012 name of same MCA ground |
| `Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium` | `Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium` | Visakhapatnam | Canonical name |
| `Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam` | `Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium` | Visakhapatnam | City suffix removed |
| `Himachal Pradesh Cricket Association Stadium` | `Himachal Pradesh Cricket Association Stadium` | Dharamsala | Canonical name |
| `Himachal Pradesh Cricket Association Stadium, Dharamsala` | `Himachal Pradesh Cricket Association Stadium` | Dharamsala | City suffix removed |
| `Sawai Mansingh Stadium` | `Sawai Mansingh Stadium` | Jaipur | Canonical name |
| `Sawai Mansingh Stadium, Jaipur` | `Sawai Mansingh Stadium` | Jaipur | City suffix removed |
| `Shaheed Veer Narayan Singh International Stadium` | `Shaheed Veer Narayan Singh International Stadium` | Raipur | Canonical name |
| `Shaheed Veer Narayan Singh International Stadium, Raipur` | `Shaheed Veer Narayan Singh International Stadium` | Raipur | City suffix removed |
| `Sheikh Zayed Stadium` | `Sheikh Zayed Stadium` | Abu Dhabi | Canonical name |
| `Zayed Cricket Stadium, Abu Dhabi` | `Sheikh Zayed Stadium` | Abu Dhabi | Alias for Sheikh Zayed Stadium |
| `Dubai International Cricket Stadium` | `Dubai International Cricket Stadium` | **Dubai** *(Imputed)* | 33 matches missing city imputed as `Dubai` |
| `Sharjah Cricket Stadium` | `Sharjah Cricket Stadium` | **Sharjah** *(Imputed)* | 18 matches missing city imputed as `Sharjah` |
| `Barabati Stadium` | `Barabati Stadium` | Cuttack | Single consistent entry |
| `Barsapara Cricket Stadium, Guwahati` | `Barsapara Cricket Stadium` | Guwahati | City suffix removed |
| `Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow` | `Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium` | Lucknow | Canonical name |
| `Buffalo Park` | `Buffalo Park` | East London | South Africa (2009) |
| `De Beers Diamond Oval` | `De Beers Diamond Oval` | Kimberley | South Africa (2009) |
| `Green Park` | `Green Park` | Kanpur | Consistent entry |
| `Holkar Cricket Stadium` | `Holkar Cricket Stadium` | Indore | Consistent entry |
| `JSCA International Stadium Complex` | `JSCA International Stadium Complex` | Ranchi | Consistent entry |
| `Kingsmead` | `Kingsmead` | Durban | South Africa (2009) |
| `Nehru Stadium` | `Nehru Stadium` | Kochi | 2011 season |
| `New Wanderers Stadium` | `New Wanderers Stadium` | Johannesburg | South Africa (2009) |
| `Newlands` | `Newlands` | Cape Town | South Africa (2009) |
| `OUTsurance Oval` | `OUTsurance Oval` | Bloemfontein | South Africa (2009) |
| `Saurashtra Cricket Association Stadium` | `Saurashtra Cricket Association Stadium` | Rajkot | Consistent entry |
| `St George's Park` | `St George's Park` | Port Elizabeth | South Africa (2009) |
| `SuperSport Park` | `SuperSport Park` | Centurion | South Africa (2009) |
| `Vidarbha Cricket Association Stadium, Jamtha` | `Vidarbha Cricket Association Stadium, Jamtha` | Nagpur | Locality suffix |

---

## 3. Missing City Resolution Summary

Across all 1,243 match-info files, exactly **51 matches** lacked a recorded `city` attribute in raw Cricsheet data:
- **33 matches** at `Dubai International Cricket Stadium` $\rightarrow$ Imputed as `Dubai`
- **18 matches** at `Sharjah Cricket Stadium` $\rightarrow$ Imputed as `Sharjah`
- **Remaining missing cities after clean preprocessing:** **0** (100% complete).
