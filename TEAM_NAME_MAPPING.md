# Canonical IPL Team Name Mapping

This document details the standardized canonical team name mappings applied to the Cricsheet IPL dataset.

---

## 1. Mapping Rationale and Franchise Boundary Rules

To ensure statistical integrity for machine learning and historical analysis, teams that underwent official rebrandings or cosmetic naming modifications are mapped to their modern canonical identities. 

> [!IMPORTANT]
> **Franchise Independence Rule:**
> Historically distinct franchises are **strictly kept separate**. 
> - **Deccan Chargers $\neq$ Sunrisers Hyderabad:** Deccan Chargers was terminated by the BCCI in 2012, and Sun TV Network won a brand new franchise bid in 2013 for Sunrisers Hyderabad. They are treated as distinct franchises.
> - **Defunct Franchises:** Pune Warriors India, Gujarat Lions, and Kochi Tuskers Kerala remain separate historical entities.

---

## 2. Team Name Mapping Table

| # | Raw Team Name in Dataset | Canonical Standardized Name | Classification | Rationale / Official Change |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Delhi Daredevils** | `Delhi Capitals` | Rebranded | Officially rebranded to Delhi Capitals in December 2018. Same franchise ownership (GMR Group / JSW Group). |
| 2 | **Delhi Capitals** | `Delhi Capitals` | Active Canonical | Current franchise name (2019–present). |
| 3 | **Kings XI Punjab** | `Punjab Kings` | Rebranded | Officially rebranded to Punjab Kings in February 2021 prior to IPL 2021. Same ownership. |
| 4 | **Punjab Kings** | `Punjab Kings` | Active Canonical | Current franchise name (2021–present). |
| 5 | **Royal Challengers Bangalore** | `Royal Challengers Bengaluru` | Rebranded | Officially updated spelling to Kannada/local Bengaluru in March 2024. Same franchise. |
| 6 | **Royal Challengers Bengaluru** | `Royal Challengers Bengaluru` | Active Canonical | Current franchise name (2024–present). |
| 7 | **Rising Pune Supergiants** | `Rising Pune Supergiant` | Rebranded | Used plural form in 2016 season; changed to singular form in 2017 season. Same franchise. |
| 8 | **Rising Pune Supergiant** | `Rising Pune Supergiant` | Defunct Canonical | Singular name used in 2017 final appearance. Franchise disbanded after 2-year tenure. |
| 9 | **Chennai Super Kings** | `Chennai Super Kings` | Active Canonical | Consistent since inaugural 2008 season. |
| 10 | **Mumbai Indians** | `Mumbai Indians` | Active Canonical | Consistent since inaugural 2008 season. |
| 11 | **Kolkata Knight Riders** | `Kolkata Knight Riders` | Active Canonical | Consistent since inaugural 2008 season. |
| 12 | **Rajasthan Royals** | `Rajasthan Royals` | Active Canonical | Consistent since inaugural 2008 season. |
| 13 | **Sunrisers Hyderabad** | `Sunrisers Hyderabad` | Active Canonical | Consistent since debut 2013 season. |
| 14 | **Gujarat Titans** | `Gujarat Titans` | Active Canonical | Consistent since debut 2022 season. |
| 15 | **Lucknow Super Giants** | `Lucknow Super Giants` | Active Canonical | Consistent since debut 2022 season. |
| 16 | **Deccan Chargers** | `Deccan Chargers` | Defunct Canonical | Defunct franchise (2008–2012). Kept separate from Sunrisers Hyderabad. |
| 17 | **Pune Warriors** | `Pune Warriors` | Defunct Canonical | Defunct franchise (2011–2013; Sahara Pune Warriors). |
| 18 | **Gujarat Lions** | `Gujarat Lions` | Defunct Canonical | Temporary 2-season franchise (2016–2017; Intex Technologies). |
| 19 | **Kochi Tuskers Kerala** | `Kochi Tuskers Kerala` | Defunct Canonical | Defunct franchise (2011 season only). |

---

## 3. Summary of Franchise Counts

- **Raw Team Names:** 19
- **Standardized Canonical Teams:** 15
- **Currently Active Franchises (10 teams):**
  1. Chennai Super Kings
  2. Delhi Capitals
  3. Gujarat Titans
  4. Kolkata Knight Riders
  5. Lucknow Super Giants
  6. Mumbai Indians
  7. Punjab Kings
  8. Rajasthan Royals
  9. Royal Challengers Bengaluru
  10. Sunrisers Hyderabad
- **Defunct / Historical Franchises (5 teams):**
  1. Deccan Chargers (2008–2012)
  2. Gujarat Lions (2016–2017)
  3. Kochi Tuskers Kerala (2011)
  4. Pune Warriors (2011–2013)
  5. Rising Pune Supergiant (2016–2017)
