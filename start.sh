#!/bin/bash

JSON_FILE="config.json"

echo "Select the option you want to execute:"
echo "1) Start the application with Gunicorn"
echo "2) Run custom bash script"
# echo "3) Both"

read -p "Enter the option number: " option

case $option in
    1)
        echo "Starting the application with Gunicorn..."
        exec gunicorn --bind 0.0.0.0:5003 index:app
        ;;
    2)
        echo "Running the bash script..."
        # exec python run-jind-multi --config "/app/config.json"
        # Modify to receive parameters from config.json PENDING
        echo "a) Modify Model Configuration"
        echo "b) Run the model with default configuration"

        read -p "Enter the selected option: " selected

        case $selected in 
            a)
                echo "The information to modify the model configuration can be found in the github repository" 
                read -p "Enter new value for PATH (app/name_of_file_to_process): " PATH
                read -p "Enter new value for BATCH_COL: " BATCH_COL
                read -p "Enter new value for LABELS_COL : " LABELS_COL
                read -p "Enter new value for SOURCE_DATASET_NAME : "  SOURCE_DATASET_NAME
                read -p "Enter new value for TARGET_DATASET_NAME : " TARGET_DATASET_NAME
                read -p "Enter new value for INTER_DATASETS_NAMES : " INTER_DATASETS_NAMES
                read -p "Enter new value for EXCLUDE_DATASETS_NAMES : " EXCLUDE_DATASETS_NAMES
                read -p "Enter new value for MIN_CELL_TYPE_POPULATION : " MIN_CELL_TYPE_POPULATION

               jq --arg path "$path" \
                  --arg batch_col "$batch_col" \
                  --arg labels_col "$labels_col" \
                  --arg source_dataset_name "$source_dataset_name" \
                  --arg target_dataset_name "$target_dataset_name" \
                  --arg inter_datasets_names "$inter_datasets_names" \
                  --argjson min_cell_type_population "$min_cell_type_population" \
                '.PATH = $path |
                    .BATCH_COL = $batch_col |
                    .LABELS_COL = $labels_col |
                    .SOURCE_DATASET_NAME = $source_dataset_name |
                    .TARGET_DATASET_NAME = $target_dataset_name |
                    .INTER_DATASETS_NAMES = $inter_datasets_names |
                    .MIN_CELL_TYPE_POPULATION = $min_cell_type_population' | "$JSON_FILE" > tmp.$$.json && mv tmp.$$.json "$JSON_FILE"

                echo "JSON fields have been updated in $JSON_FILE"
				;;
                
            b)
                read -p "Enter new value for PATH (app/name_of_file_to_process): " PATH
                jq --arg value "$path" '.model_configuration.PATH = $path' "$JSON_FILE" > tmp.$$.json && mv tmp.$$.json "$JSON_FILE"

                echo "BATCH_COL has been updated in $JSON_FILE"
				;;
		esac
        exec python run-jind-multi --config "config.json"
        ;;
    *)
        echo "Invalid option."
        ;;
esac
