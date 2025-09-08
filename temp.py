import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Spotify data
# Option 1: Use the full file path
df = pd.read_csv(r'C:\Users\jacob\Desktop\Anaconda\Data_Visualization_And_Modeling-main\Data_Visualization_And_Modeling-main\Lab\spotify-2023.csv', encoding='ISO-8859-1')

# Option 2: If you want to change the working directory first (alternative approach)
# import os
# os.chdir(r'C:\Users\jacob\Desktop\Anaconda\Data_Visualization_And_Modeling-main\Data_Visualization_And_Modeling-main\Lab')
# df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# =============================================================================
# STEP 1: VISUALIZE AND DESCRIBE DISTRIBUTIONS
# =============================================================================

# Set up the plotting style
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# 1. BPM Distribution
plt.subplot(3, 3, 1)
plt.hist(df['bpm'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('BPM Distribution')
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Calculate BPM statistics
bpm_mean = df['bpm'].mean()
bpm_median = df['bpm'].median()
bpm_mode = df['bpm'].mode()[0]
print(f"\nBPM Statistics:")
print(f"Mean: {bpm_mean:.1f}")
print(f"Median: {bpm_median:.1f}")
print(f"Mode: {bpm_mode}")
print(f"Most common range: {bpm_median-10:.0f}-{bpm_median+10:.0f}")

# 2. Key Distribution
plt.subplot(3, 3, 2)
key_counts = df['key'].value_counts()
key_counts.plot(kind='bar', color='lightcoral')
plt.title('Key Distribution')
plt.xlabel('Musical Key')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

most_common_key = key_counts.index[0]
print(f"\nKey Statistics:")
print(f"Most common key: {most_common_key}")
print(f"Key frequencies:\n{key_counts}")

# 3. Mode Distribution
plt.subplot(3, 3, 3)
mode_counts = df['mode'].value_counts()
colors = ['gold', 'lightblue']
mode_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Mode Distribution')
plt.ylabel('')

most_common_mode = mode_counts.index[0]
print(f"\nMode Statistics:")
print(f"Most common mode: {most_common_mode}")
print(f"Mode percentages:\n{mode_counts/len(df)*100}")

# 4. Danceability Distribution
plt.subplot(3, 3, 4)
plt.hist(df['danceability_%'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
plt.title('Danceability Distribution')
plt.xlabel('Danceability %')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

dance_mean = df['danceability_%'].mean()
dance_median = df['danceability_%'].median()
print(f"\nDanceability Statistics:")
print(f"Mean: {dance_mean:.1f}%")
print(f"Median: {dance_median:.1f}%")

# 5. Energy Distribution
plt.subplot(3, 3, 5)
plt.hist(df['energy_%'], bins=20, color='orange', alpha=0.7, edgecolor='black')
plt.title('Energy Distribution')
plt.xlabel('Energy %')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

energy_mean = df['energy_%'].mean()
energy_median = df['energy_%'].median()
print(f"\nEnergy Statistics:")
print(f"Mean: {energy_mean:.1f}%")
print(f"Median: {energy_median:.1f}%")

# 6. Speechiness Distribution
plt.subplot(3, 3, 6)
plt.hist(df['speechiness_%'], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.title('Speechiness Distribution')
plt.xlabel('Speechiness %')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

speech_mean = df['speechiness_%'].mean()
speech_median = df['speechiness_%'].median()
print(f"\nSpeechiness Statistics:")
print(f"Mean: {speech_mean:.1f}%")
print(f"Median: {speech_median:.1f}%")

# 7. Acousticness Distribution
plt.subplot(3, 3, 7)
plt.hist(df['acousticness_%'], bins=20, color='brown', alpha=0.7, edgecolor='black')
plt.title('Acousticness Distribution')
plt.xlabel('Acousticness %')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

acoustic_mean = df['acousticness_%'].mean()
acoustic_median = df['acousticness_%'].median()
print(f"\nAcousticness Statistics:")
print(f"Mean: {acoustic_mean:.1f}%")
print(f"Median: {acoustic_median:.1f}%")

plt.tight_layout()
plt.show()

# =============================================================================
# STEP 2: DETERMINE "GUARANTEED HIT" CRITERIA
# =============================================================================

print("\n" + "="*60)
print("GUARANTEED SMASH HIT FORMULA")
print("="*60)

# Based on the most common/typical values, let's define our hit criteria
hit_criteria = {
    'bpm_min': bpm_median - 15,  # Range around median
    'bpm_max': bpm_median + 15,
    'key': most_common_key,      # Most common key
    'mode': most_common_mode,    # Most common mode
    'danceability_min': dance_median - 10,  # Range around median
    'danceability_max': dance_median + 10,
    'energy_min': energy_median - 10,  # Range around median
    'energy_max': energy_median + 10,
    'speechiness_max': speech_median + 5,  # Low speechiness (not too much rap)
    'acousticness_max': acoustic_median + 10  # Not too acoustic
}

print(f"Optimal BPM: {hit_criteria['bpm_min']:.0f}-{hit_criteria['bpm_max']:.0f}")
print(f"Optimal Key: {hit_criteria['key']}")
print(f"Optimal Mode: {hit_criteria['mode']}")
print(f"Optimal Danceability: {hit_criteria['danceability_min']:.0f}-{hit_criteria['danceability_max']:.0f}%")
print(f"Optimal Energy: {hit_criteria['energy_min']:.0f}-{hit_criteria['energy_max']:.0f}%")
print(f"Optimal Speechiness: ≤ {hit_criteria['speechiness_max']:.0f}%")
print(f"Optimal Acousticness: ≤ {hit_criteria['acousticness_max']:.0f}%")

# =============================================================================
# STEP 3: FIND SONGS THAT MEET ALL CRITERIA
# =============================================================================

print("\n" + "="*60)
print("FINDING SONGS THAT MEET ALL CRITERIA")
print("="*60)

# Filter songs based on our criteria
hit_songs = df[
    (df['bpm'] >= hit_criteria['bpm_min']) & 
    (df['bpm'] <= hit_criteria['bpm_max']) &
    (df['key'] == hit_criteria['key']) &
    (df['mode'] == hit_criteria['mode']) &
    (df['danceability_%'] >= hit_criteria['danceability_min']) &
    (df['danceability_%'] <= hit_criteria['danceability_max']) &
    (df['energy_%'] >= hit_criteria['energy_min']) &
    (df['energy_%'] <= hit_criteria['energy_max']) &
    (df['speechiness_%'] <= hit_criteria['speechiness_max']) &
    (df['acousticness_%'] <= hit_criteria['acousticness_max'])
]

print(f"Number of songs meeting all criteria: {len(hit_songs)}")

if len(hit_songs) > 0:
    print("\nSONGS THAT MEET ALL CRITERIA:")
    print("-" * 50)
    
    # First, let's clean the streams column (remove commas, convert to numeric)
    df['streams_clean'] = pd.to_numeric(df['streams'].astype(str).str.replace(',', ''), errors='coerce')
    hit_songs = hit_songs.copy()
    hit_songs['streams_clean'] = pd.to_numeric(hit_songs['streams'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Sort by streams (popularity) - FIXED: removed na_last parameter
    hit_songs_sorted = hit_songs.sort_values('streams_clean', ascending=False)
    
    for idx, song in hit_songs_sorted.iterrows():
        print(f"Artist: {song['artist(s)_name']}")
        print(f"Title: {song['track_name']}")
        print(f"Streams: {song['streams']}")
        print(f"Released: {song['released_year']}")
        print(f"BPM: {song['bpm']}, Key: {song['key']}, Mode: {song['mode']}")
        print(f"Danceability: {song['danceability_%']}%, Energy: {song['energy_%']}%")
        print(f"Speechiness: {song['speechiness_%']}%, Acousticness: {song['acousticness_%']}%")
        print("-" * 50)
    
    # Check if any of these were actually popular
    print(f"\nPOPULARITY ANALYSIS:")
    print(f"Average streams of 'hit formula' songs: {hit_songs['streams_clean'].mean():,.0f}")
    print(f"Average streams of all songs: {df['streams_clean'].mean():,.0f}")
    
    # Check if our formula songs are in top percentiles
    top_10_percent_threshold = df['streams_clean'].quantile(0.9)
    formula_in_top_10 = hit_songs[hit_songs['streams_clean'] >= top_10_percent_threshold]
    print(f"Number of 'formula' songs in top 10% by streams: {len(formula_in_top_10)}")
    
else:
    print("No songs meet ALL criteria. Let's relax some constraints...")
    
    # Try with relaxed criteria
    relaxed_hit_songs = df[
        (df['bpm'] >= hit_criteria['bmp_min']) & 
        (df['bpm'] <= hit_criteria['bpm_max']) &
        (df['key'] == hit_criteria['key']) &
        (df['mode'] == hit_criteria['mode'])
    ]
    
    print(f"Songs meeting basic criteria (BPM, Key, Mode): {len(relaxed_hit_songs)}")
    
    if len(relaxed_hit_songs) > 0:
        print("\nTop 5 songs with basic criteria:")
        relaxed_sorted = relaxed_hit_songs.sort_values('streams_clean', ascending=False).head()
        for idx, song in relaxed_sorted.iterrows():
            print(f"{song['artist(s)_name']} - {song['track_name']}")

# =============================================================================
# STEP 4: ANALYZE A KNOWN HIT SONG
# =============================================================================

print("\n" + "="*60)
print("ANALYZING A KNOWN HIT SONG")
print("="*60)

# Let's find a very popular song and see how it compares to our formula
most_streamed = df.loc[df['streams_clean'].idxmax()]

print("Most streamed song in dataset:")
print(f"Artist: {most_streamed['artist(s)_name']}")
print(f"Title: {most_streamed['track_name']}")
print(f"Streams: {most_streamed['streams']}")
print("\nHow it compares to our 'formula':")
print(f"BPM: {most_streamed['bpm']} (our formula: {hit_criteria['bpm_min']:.0f}-{hit_criteria['bpm_max']:.0f})")
print(f"Key: {most_streamed['key']} (our formula: {hit_criteria['key']})")
print(f"Mode: {most_streamed['mode']} (our formula: {hit_criteria['mode']})")
print(f"Danceability: {most_streamed['danceability_%']}% (our formula: {hit_criteria['danceability_min']:.0f}-{hit_criteria['danceability_max']:.0f}%)")
print(f"Energy: {most_streamed['energy_%']}% (our formula: {hit_criteria['energy_min']:.0f}-{hit_criteria['energy_max']:.0f}%)")
print(f"Speechiness: {most_streamed['speechiness_%']}% (our formula: ≤{hit_criteria['speechiness_max']:.0f}%)")
print(f"Acousticness: {most_streamed['acousticness_%']}% (our formula: ≤{hit_criteria['acousticness_max']:.0f}%)")

# Check how many criteria the most popular song meets
criteria_met = 0
if hit_criteria['bpm_min'] <= most_streamed['bpm'] <= hit_criteria['bpm_max']:
    criteria_met += 1
if most_streamed['key'] == hit_criteria['key']:
    criteria_met += 1
if most_streamed['mode'] == hit_criteria['mode']:
    criteria_met += 1
if hit_criteria['danceability_min'] <= most_streamed['danceability_%'] <= hit_criteria['danceability_max']:
    criteria_met += 1
if hit_criteria['energy_min'] <= most_streamed['energy_%'] <= hit_criteria['energy_max']:
    criteria_met += 1
if most_streamed['speechiness_%'] <= hit_criteria['speechiness_max']:
    criteria_met += 1
if most_streamed['acousticness_%'] <= hit_criteria['acousticness_max']:
    criteria_met += 1

print(f"\nThe most popular song meets {criteria_met}/7 of our formula criteria.")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("The 'guaranteed hit formula' based on most common characteristics")
print("may not actually guarantee a hit! This suggests that:")
print("1. Musical success involves more than just common characteristics")
print("2. Uniqueness and creativity might be more important than following formulas")
print("3. Other factors like marketing, timing, and artist popularity matter")
print("4. Data-driven approaches have limitations in creative fields")
