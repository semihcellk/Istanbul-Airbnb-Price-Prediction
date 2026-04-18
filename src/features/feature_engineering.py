import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.cluster import KMeans
import warnings
import os

"""
AI USAGE DECLARATION:
- Syntax Support: Used AI to generate the correct Regex and 'ast.literal_eval' syntax for parsing the 'amenities' column.
- Implementation: The 'haversine_distance' mathematical formula implementation was assisted by AI tools.
- Refactoring: Modular structure of 'load_calendar_features' was refined with AI suggestions.
"""

DEMO_MODE = False

warnings.filterwarnings("ignore")

# PATHS
TRAIN_INPUT_PATH = 'data/interim/filled_cleaned_train.csv'
TEST_INPUT_PATH = 'data/test.csv'
CALENDAR_PATH = 'data/calendar.csv'
REVIEWS_PATH = 'data/reviews.csv'
TRAIN_OUTPUT_PATH = 'data/processed/train/engineered_train.csv'
TEST_OUTPUT_PATH = 'data/processed/test/engineered_test.csv'

# CONSTANTS
COORDS = {
    'Taksim': (41.037, 28.985),
    'Sultanahmet': (41.005, 28.976),
    'Kadikoy': (40.991, 29.027),
    'Besiktas': (41.042, 29.006),
    'Airport': (41.281, 28.751),
    'Galata': (41.026, 28.974),
    'Ortakoy': (41.055, 29.027)
}
REFERENCE_DATE = pd.to_datetime('2025-06-27')

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth using the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of the first point (in degrees).
        lat2, lon2: Latitude and longitude of the second point (in degrees).

    Returns:
        Distance in kilometers.
    """
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def load_calendar_features():
    """Extract temporal features from the Kaggle calendar.csv file.

    Computes availability ratios, min/max night statistics, consecutive
    availability blocks, and availability change frequency per listing.

    Returns:
        pd.DataFrame: Calendar-derived features indexed by listing id.
    """
    calendar_df = pd.read_csv(CALENDAR_PATH)
    
    calendar_features = calendar_df.groupby('listing_id').agg({
        'available': lambda x: (x == 't').sum() / len(x) if len(x) > 0 else 0,
        'minimum_nights': ['mean', 'std', 'min', 'max'],
        'maximum_nights': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    calendar_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                  for col in calendar_features.columns]
    calendar_features.rename(columns={'listing_id': 'id'}, inplace=True)
    
    # Additional calendar features
    calendar_df['available_binary'] = (calendar_df['available'] == 't').astype(int)
    
    # Consecutive availability blocks
    availability_stats = calendar_df.sort_values(['listing_id', 'date']).groupby('listing_id').apply(
        lambda x: pd.Series({
            'max_consecutive_available': x['available_binary'].groupby(
                (x['available_binary'] != x['available_binary'].shift()).cumsum()
            ).sum().max(),
            'availability_changes': (x['available_binary'] != x['available_binary'].shift()).sum()
        }), include_groups=False
    ).reset_index()
    
    calendar_features = calendar_features.merge(availability_stats, left_on='id', right_on='listing_id', how='left')
    calendar_features.drop(columns=['listing_id'], inplace=True, errors='ignore')
    
    print(f"Calendar features shape: {calendar_features.shape}")
    return calendar_features

def load_review_features():
    """Extract review-based features from the Kaggle reviews.csv file.

    Computes total review counts, recency windows (30/90/180/365 days),
    time-since-first/last review, average inter-review gaps, and
    review frequency per listing.

    Returns:
        pd.DataFrame: Review-derived features indexed by listing id.
    """
    reviews_df = pd.read_csv(REVIEWS_PATH)
    reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')
    
    # Total review count
    review_counts = reviews_df.groupby('listing_id').size().reset_index(name='total_reviews_from_file')
    
    # Days since last review
    reviews_df['days_ago'] = (REFERENCE_DATE - reviews_df['date']).dt.days
    
    # Recent review counts
    recent_reviews = reviews_df.groupby('listing_id').apply(lambda x: pd.Series({
        'reviews_last_30d': (x['days_ago'] <= 30).sum(),
        'reviews_last_90d': (x['days_ago'] <= 90).sum(),
        'reviews_last_180d': (x['days_ago'] <= 180).sum(),
        'reviews_last_365d': (x['days_ago'] <= 365).sum(),
        'days_since_last_review': x['days_ago'].min() if len(x) > 0 else np.nan,
        'days_since_first_review': x['days_ago'].max() if len(x) > 0 else np.nan,
        'avg_days_between_reviews': x['days_ago'].diff().abs().mean() if len(x) > 1 else np.nan,
        'review_frequency': len(x) / (x['days_ago'].max() + 1) if len(x) > 0 and x['days_ago'].max() > 0 else 0
    }), include_groups=False).reset_index()
    
    # Merge all review features
    review_features = review_counts.merge(recent_reviews, on='listing_id', how='left')
    review_features.rename(columns={'listing_id': 'id'}, inplace=True)
    
    print(f"Review features shape: {review_features.shape}")
    return review_features

def engineer_features(dataframe, is_train=True, kmeans_model=None, neigh_stats=None):
    """Apply all feature engineering transformations to a train or test DataFrame.

    Creates date, amenity, location/clustering, ratio, host quality,
    review score, interaction, text mining, neighbourhood statistics,
    and availability/booking features.

    Args:
        dataframe: Raw input DataFrame.
        is_train: Whether this is training data (used for neighbourhood stats).
        kmeans_model: Pre-fitted KMeans model (None for train, reused for test).
        neigh_stats: Pre-computed neighbourhood statistics (None for train).

    Returns:
        Tuple of (engineered DataFrame, KMeans model, neighbourhood stats).
    """
    data = dataframe.copy()
    
    print(f"Starting feature engineering for {'train' if is_train else 'test'}...")
    
    # DATE FEATURES
    if 'host_since' in data.columns:
        data['host_since'] = pd.to_datetime(data['host_since'], errors='coerce')
        data['host_days_active'] = (REFERENCE_DATE - data['host_since']).dt.days
        data['host_days_active'] = data['host_days_active'].fillna(data['host_days_active'].median())
        data['host_years_active'] = data['host_days_active'] / 365.25
    else:
        data['host_days_active'] = 0
        data['host_years_active'] = 0
    
    # AMENITIES & TEXT
    if 'amenities' in data.columns:
        def count_amenities(x):
            try: 
                return len(ast.literal_eval(x.replace('{', '[').replace('}', ']')))
            except (ValueError, SyntaxError): 
                return 0
        
        data['amenity_count'] = data['amenities'].astype(str).apply(count_amenities)
        
        amenities_str = data['amenities'].astype(str)
        
        # Basic amenities
        basic_keywords = ['Wifi', 'Air conditioning', 'Heating', 'Kitchen', 'Washer', 
                         'Dryer', 'TV', 'Hair dryer', 'Iron']
        for k in basic_keywords:
            data[f'has_{k.lower().replace(" ", "_")}'] = amenities_str.str.contains(k, case=False).astype(int)
        
        # Luxury amenities
        luxury_keywords = ['Pool', 'Jacuzzi', 'Gym', 'Hot tub', 'Sauna', 'BBQ grill']
        for k in luxury_keywords:
            data[f'has_{k.lower().replace(" ", "_")}'] = amenities_str.str.contains(k, case=False).astype(int)
        
        # View/Space amenities
        space_keywords = ['Balcony', 'Patio', 'Garden', 'Terrace', 'Waterfront']
        for k in space_keywords:
            data[f'has_{k.lower().replace(" ", "_")}'] = amenities_str.str.contains(k, case=False).astype(int)
        
        # Count luxury amenities
        data['luxury_amenity_count'] = sum(data[f'has_{k.lower().replace(" ", "_")}'] 
                                           for k in luxury_keywords)
        data['space_amenity_count'] = sum(data[f'has_{k.lower().replace(" ", "_")}'] 
                                          for k in space_keywords)
    else:
        data['amenity_count'] = 0
        data['luxury_amenity_count'] = 0
        data['space_amenity_count'] = 0
    
    # LOCATION & CLUSTERING
    if 'latitude' in data.columns and 'longitude' in data.columns:
        # Distance to key locations
        for name, (lat, lon) in COORDS.items():
            data[f'dist_to_{name.lower()}'] = haversine_distance(
                data['latitude'], data['longitude'], lat, lon
            )
        
        # Minimum distance to any major location
        dist_cols = [f'dist_to_{name.lower()}' for name in COORDS.keys()]
        data['min_dist_to_center'] = data[dist_cols].min(axis=1)
        
        # Clustering
        loc_data = data[['latitude', 'longitude']].fillna(0)
        if kmeans_model is None:
            kmeans_model = KMeans(n_clusters=30, random_state=42, n_init=10)
            data['loc_cluster'] = kmeans_model.fit_predict(loc_data)
        else:
            data['loc_cluster'] = kmeans_model.predict(loc_data)
    
    # RATIO FEATURES
    data['bedroom_per_person'] = data['bedrooms'] / (data['accommodates'] + 1)
    data['bathroom_per_person'] = data['bathrooms'] / (data['accommodates'] + 1)
    data['beds_per_bedroom'] = data['beds'] / (data['bedrooms'] + 1)
    data['beds_per_person'] = data['beds'] / (data['accommodates'] + 1)
    
    # Space efficiency
    data['total_rooms'] = data['bedrooms'] + data['bathrooms']
    data['rooms_per_person'] = data['total_rooms'] / (data['accommodates'] + 1)
    
    # HOST QUALITY FEATURES
    # Host quality score
    data['host_is_superhost_binary'] = data['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
    data['host_identity_verified_binary'] = data['host_identity_verified'].map({'t': 1, 'f': 0}).fillna(0)
    data['host_has_profile_pic_binary'] = data['host_has_profile_pic'].map({'t': 1, 'f': 0}).fillna(0)
    
    data['host_quality_score'] = (
        data['host_is_superhost_binary'] * 0.4 +
        data['host_identity_verified_binary'] * 0.3 +
        data['host_has_profile_pic_binary'] * 0.3
    )
    
    # Professional host indicator
    data['host_is_professional'] = (data['host_total_listings_count'] > 5).astype(int)
    data['host_listing_density'] = data['host_total_listings_count'] / (data['host_days_active'] + 1)
    
    # Instant bookable
    data['is_instant_bookable'] = data['instant_bookable'].map({'t': 1, 'f': 0}).fillna(0)
    
    # REVIEW SCORE FEATURES
    review_cols = ['review_scores_rating', 'review_scores_accuracy', 
                   'review_scores_cleanliness', 'review_scores_checkin',
                   'review_scores_communication', 'review_scores_location', 
                   'review_scores_value']
    
    # Overall composite score
    data['review_score_overall'] = (
        data['review_scores_rating'].fillna(0) * 0.3 +
        data['review_scores_cleanliness'].fillna(0) * 0.25 +
        data['review_scores_location'].fillna(0) * 0.25 +
        data['review_scores_value'].fillna(0) * 0.2
    )
    
    # Variance and range
    data['review_score_variance'] = data[review_cols].var(axis=1, skipna=True)
    data['review_score_min'] = data[review_cols].min(axis=1, skipna=True)
    data['review_score_max'] = data[review_cols].max(axis=1, skipna=True)
    data['review_score_range'] = data['review_score_max'] - data['review_score_min']
    
    # Missing review scores indicator
    data['missing_review_scores'] = data[review_cols].isnull().sum(axis=1)
    if 'total_reviews_from_file' in data.columns:
        data['has_reviews'] = (data['total_reviews_from_file'] > 0).astype(int)
    else:
        data['has_reviews'] = 0
    
    # 7. INTERACTION FEATURES
    data['loc_room_interaction'] = (
        data['neighbourhood_cleansed'].astype(str) + "_" + 
        data['room_type'].astype(str)
    )
    
    data['prop_neigh_interaction'] = (
        data['property_type'].astype(str) + "_" + 
        data['neighbourhood_cleansed'].astype(str)
    )
    
    data['room_prop_interaction'] = (
        data['room_type'].astype(str) + "_" + 
        data['property_type'].astype(str)
    )
    
    # TEXT MINING
    # Combine text fields
    data['text_features'] = (
        data['name'].fillna('').astype(str) + " " + 
        data['description'].fillna('').astype(str) + " " +
        data['neighborhood_overview'].fillna('').astype(str)
    )
    data['text_features'] = data['text_features'].str.lower()
    
    # Text length features
    data['desc_len'] = data['description'].astype(str).apply(len)
    data['name_len'] = data['name'].astype(str).apply(len)
    data['name_word_count'] = data['name'].astype(str).str.split().str.len()
    
    # Luxury keywords
    luxury_keywords = ['luxury', 'premium', 'deluxe', 'exclusive', 'elegant', 
                      'sophisticated', 'upscale', 'high-end']
    data['luxury_keyword_count'] = 0
    for kw in luxury_keywords:
        data['luxury_keyword_count'] += data['text_features'].str.contains(kw, na=False).astype(int)
    
    # Location keywords
    location_keywords = ['bosphorus', 'sea view', 'waterfront', 'center', 'central',
                        'metro', 'taksim', 'sultanahmet', 'historic', 'view', 
                        'panoramic', 'galata', 'karakoy']
    data['location_keyword_count'] = 0
    for kw in location_keywords:
        data['location_keyword_count'] += data['text_features'].str.contains(kw, na=False).astype(int)
    
    # Amenity keywords in text
    amenity_text_keywords = ['pool', 'gym', 'spa', 'jacuzzi', 'terrace', 
                             'balcony', 'garden', 'parking', 'elevator']
    data['amenity_keyword_count'] = 0
    for kw in amenity_text_keywords:
        data['amenity_keyword_count'] += data['text_features'].str.contains(kw, na=False).astype(int)
    
    # Individual important keywords
    important_keywords = ['bosphorus', 'luxury', 'view', 'central', 'metro', 
                         'terrace', 'sea', 'historical', 'modern', 'renovated']
    for kw in important_keywords:
        data[f'txt_{kw}'] = data['text_features'].str.contains(kw, na=False).astype(int)
    
    data.drop(columns=['text_features'], inplace=True)
    
    # NEIGHBOURHOOD STATISTICS
    if is_train and 'price' in data.columns:
        # Calculate neighbourhood price statistics from training data
        neigh_stats = data.groupby('neighbourhood_cleansed')['price'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).add_prefix('neigh_price_')
        neigh_stats['neigh_price_range'] = neigh_stats['neigh_price_max'] - neigh_stats['neigh_price_min']
        neigh_stats['neigh_price_cv'] = neigh_stats['neigh_price_std'] / (neigh_stats['neigh_price_mean'] + 1)
        neigh_stats = neigh_stats.reset_index()
    
    if neigh_stats is not None:
        data = data.merge(neigh_stats, on='neighbourhood_cleansed', how='left')
        # Fill missing neighbourhood stats with global means
        neigh_cols = [c for c in neigh_stats.columns if c.startswith('neigh_price_') and c != 'neigh_price_count']
        for col in neigh_cols:
            if col in data.columns:
                global_val = data[col].mean() if data[col].notna().any() else 0
                data[col].fillna(global_val, inplace=True)
    
    # AVAILABILITY & BOOKING FEATURES
    # Minimum/maximum nights features
    data['min_nights_flexibility'] = (data['minimum_nights'] <= 2).astype(int)
    data['accepts_short_stays'] = (data['minimum_nights'] == 1).astype(int)
    data['long_stay_focused'] = (data['minimum_nights'] >= 7).astype(int)
    
    print(f"Final shape: {data.shape}")
    
    return data, kmeans_model, neigh_stats

def main():
    train_df = pd.read_csv(TRAIN_INPUT_PATH)
    test_df = pd.read_csv(TEST_INPUT_PATH)
    
    if DEMO_MODE:
        print("DEMO MODE ACTIVE")
        train_df = train_df.head(500).copy()
        test_df = test_df.head(100).copy()
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Load calendar features
    calendar_features = load_calendar_features()
    
    # Load review features
    review_features = load_review_features()
    
    # Merge calendar and review features
    train_df = train_df.merge(calendar_features, on='id', how='left')
    test_df = test_df.merge(calendar_features, on='id', how='left')
    
    train_df = train_df.merge(review_features, on='id', how='left')
    test_df = test_df.merge(review_features, on='id', how='left')
    
    # Fill NaN values from merged features
    calendar_cols = [c for c in calendar_features.columns if c != 'id']
    review_cols = [c for c in review_features.columns if c != 'id']
    
    for col in calendar_cols + review_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)
    
    print(f"\nAfter merging - Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Engineer features
    train_eng, kmeans_model, neigh_stats = engineer_features(train_df, is_train=True)
    test_eng, _, _ = engineer_features(test_df, is_train=False, 
                                        kmeans_model=kmeans_model, 
                                        neigh_stats=neigh_stats)
    
    # Save engineered data
    os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_OUTPUT_PATH), exist_ok=True)
    
    train_eng.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_eng.to_csv(TEST_OUTPUT_PATH, index=False)
    
    print(f"Train: {TRAIN_OUTPUT_PATH}")
    print(f"Test: {TEST_OUTPUT_PATH}")
    print(f"\nFinal shapes - Train: {train_eng.shape}, Test: {test_eng.shape}")

if __name__ == "__main__":
    main()