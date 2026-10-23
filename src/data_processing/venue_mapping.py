"""
Canonical Venue and City Standardization Mapping

Consolidates naming aliases and punctuation variations of identical physical grounds
to canonical venue names, while preserving distinct stadiums in the same city.
Also provides conservative city imputation for venues where city is 100% unambiguous.
"""

from typing import Optional

CANONICAL_VENUE_MAP = {
    # M Chinnaswamy Stadium (Bengaluru)
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",

    # Eden Gardens (Kolkata)
    "Eden Gardens": "Eden Gardens",
    "Eden Gardens, Kolkata": "Eden Gardens",

    # Wankhede Stadium (Mumbai)
    "Wankhede Stadium": "Wankhede Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",

    # Brabourne Stadium (Mumbai)
    "Brabourne Stadium": "Brabourne Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",

    # Dr DY Patil Sports Academy (Navi Mumbai)
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",

    # Arun Jaitley Stadium / Feroz Shah Kotla (Delhi)
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",

    # Rajiv Gandhi International Stadium (Hyderabad)
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",

    # MA Chidambaram Stadium (Chennai)
    "MA Chidambaram Stadium": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",

    # Narendra Modi Stadium / Motera (Ahmedabad)
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",

    # PCA Stadium (Mohali)
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",

    # Maharaja Yadavindra Singh Stadium (Mullanpur / New Chandigarh)
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",

    # Maharashtra Cricket Association Stadium / Subrata Roy Sahara Stadium (Pune)
    "Maharashtra Cricket Association Stadium": "Maharashtra Cricket Association Stadium",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",

    # Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium (Visakhapatnam)
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",

    # HPCA Stadium (Dharamsala)
    "Himachal Pradesh Cricket Association Stadium": "Himachal Pradesh Cricket Association Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",

    # Sawai Mansingh Stadium (Jaipur)
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",

    # Shaheed Veer Narayan Singh Stadium (Raipur)
    "Shaheed Veer Narayan Singh International Stadium": "Shaheed Veer Narayan Singh International Stadium",
    "Shaheed Veer Narayan Singh International Stadium, Raipur": "Shaheed Veer Narayan Singh International Stadium",

    # Sheikh Zayed Stadium (Abu Dhabi)
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",

    # Single-occurrence / consistent venues
    "Barabati Stadium": "Barabati Stadium",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Buffalo Park": "Buffalo Park",
    "De Beers Diamond Oval": "De Beers Diamond Oval",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
    "Green Park": "Green Park",
    "Holkar Cricket Stadium": "Holkar Cricket Stadium",
    "JSCA International Stadium Complex": "JSCA International Stadium Complex",
    "Kingsmead": "Kingsmead",
    "Nehru Stadium": "Nehru Stadium",
    "New Wanderers Stadium": "New Wanderers Stadium",
    "Newlands": "Newlands",
    "OUTsurance Oval": "OUTsurance Oval",
    "Saurashtra Cricket Association Stadium": "Saurashtra Cricket Association Stadium",
    "Sharjah Cricket Stadium": "Sharjah Cricket Stadium",
    "St George's Park": "St George's Park",
    "SuperSport Park": "SuperSport Park",
    "Vidarbha Cricket Association Stadium, Jamtha": "Vidarbha Cricket Association Stadium, Jamtha",
}

# Unambiguous venue-to-city imputation dictionary
VENUE_CITY_IMPUTATION = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
}


def map_venue_name(venue: Optional[str]) -> Optional[str]:
    """
    Map a raw venue string to its canonical standardized name.

    Args:
        venue: Raw venue name or None.

    Returns:
        Canonical venue name, or None/empty string if input is empty/None.
    """
    if not venue:
        return venue
    clean_venue = venue.strip()
    return CANONICAL_VENUE_MAP.get(clean_venue, clean_venue)


def impute_city_if_missing(venue: Optional[str], current_city: Optional[str]) -> Optional[str]:
    """
    Impute missing city only when the venue provides a 100% unambiguous city.
    Do not guess ambiguous cities.

    Args:
        venue: Venue name (either raw or canonical).
        current_city: Currently recorded city value.

    Returns:
        Imputed city or original city.
    """
    if current_city and current_city.strip():
        return current_city.strip()

    if not venue:
        return current_city

    canonical_venue = map_venue_name(venue)
    return VENUE_CITY_IMPUTATION.get(canonical_venue, current_city)
