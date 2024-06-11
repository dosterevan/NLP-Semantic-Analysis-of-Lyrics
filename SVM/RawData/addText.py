import csv
import os 

# Example usage
current_directory = os.path.dirname(os.path.realpath(__file__))

def combine_csv_files(muse_csv, spotify_csv, output_csv):
    muse_data = []
    with open(muse_csv, 'r', newline='', encoding='utf-8') as muse_file:
        muse_reader = csv.DictReader(muse_file)
        for row in muse_reader:
            muse_data.append(row)

    spotify_data = {}
    with open(spotify_csv, 'r', newline='', encoding='utf-8') as spotify_file:
        spotify_reader = csv.DictReader(spotify_file)
        for row in spotify_reader:
            spotify_data[(row['artist'], row['song'])] = row['text']

    # Create a new list with combined data
    combined_data = []
    for entry in muse_data:
        key = (entry['artist'], entry['track'])
        if key in spotify_data:
            entry['text'] = spotify_data[key]
            combined_data.append(entry)

    # Write the combined data to a new CSV file
    fieldnames = ['lastfm_url', 'track', 'artist', 'seeds', 'number_of_emotion_tags', 'valence_tags',
                  'arousal_tags', 'dominance_tags', 'mbid', 'spotify_id', 'genre', 'text']
    with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_data)

# Example usage:
muse_csv = 'muse_v3.csv'
spotify_csv = 'spotify_millsongdata.csv'
output_csv = 'combined_data.csv'
combine_csv_files(muse_csv, spotify_csv, output_csv)

# author_seed_csv_path = os.path.join(current_directory, "train/unprocessed_train.csv")
# output_csv_path = os.path.join(current_directory, "train/processed_train.csv")