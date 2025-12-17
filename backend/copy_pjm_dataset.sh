#!/bin/bash

# Destination directory in the project
DEST_DIR="translator/data/raw"

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Error: Destination directory $DEST_DIR does not exist."
    exit 1
fi

# Function to process a single source directory
process_source_dir() {
    local SOURCE_DIR="$1"
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "Error: Source directory $SOURCE_DIR does not exist."
        return 1
    fi

    # Process each letter folder (A, B, C, etc.) in the source directory
    for letter_dir in "$SOURCE_DIR"/*; do
        if [ -d "$letter_dir" ]; then
            letter=$(basename "$letter_dir")
            echo "Copying images for letter $letter from $SOURCE_DIR..."
            mkdir -p "$DEST_DIR/$letter"

            # Count existing images in the destination to avoid overwriting
            existing_count=$(ls "$DEST_DIR/$letter"/*.jpg 2>/dev/null | wc -l)

            # Copy and rename images
            i=$((existing_count + 1))
            for img_path in "$letter_dir"/*.jpg; do
                if [ -f "$img_path" ]; then
                    new_name="$DEST_DIR/$letter/${letter}${i}.jpg"
                    cp "$img_path" "$new_name"
                    echo "Copied $img_path to $new_name"
                    i=$((i + 1))
                fi
            done
        fi
    done
}

# Process multiple source directories
process_source_dir "$HOME/Zdjęcia"
process_source_dir "$HOME/znaki2"

echo "Dataset copied successfully to $DEST_DIR."
