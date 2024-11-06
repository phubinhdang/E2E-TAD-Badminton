
n to display usage
usage() {
	    echo "Usage: $0 --num_queries '2 4 8' --epochs 5"
	        echo "Example: $0 --num_queries '2 4 8' --epochs 5"
		    exit 1
	    }

	    # Default values
	    NUM_QUERIES_VALUES=()
	    EPOCHS_VALUE=""

	    # Parse command-line arguments
	    while [[ "$#" -gt 0 ]]; do
		        case $1 in
				        --num_queries) 
						            shift
							                NUM_QUERIES_VALUES=($1) 
									            ;;
										            --epochs) 
												                shift
														            EPOCHS_VALUE=$1 
															                ;;
																	        *) 
																			            usage 
																				                ;;
																						    esac
																						        shift
																						done

																						# Check if required parameters are provided
																						if [ ${#NUM_QUERIES_VALUES[@]} -eq 0 ] || [ -z "$EPOCHS_VALUE" ]; then
																							    usage
																						    fi

																						    # Define the base configuration file and the Docker image name
																						    CONFIG_FILE="configs/badminton_e2e_slowfast_tadtr.yml"
																						    IMAGE_NAME="tad_badminton"

																						    # Loop through each value in NUM_QUERIES_VALUES
																						    for VALUE in "${NUM_QUERIES_VALUES[@]}"; do
																							        # Create a backup of the original config file
																								    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"
																								        
																								        # Modify the num_queries, output_dir, and epochs values in the config file
																									    sed -E "s/(^num_queries: ).*/\1$VALUE/" "$CONFIG_FILE" | \
																										        sed -E "s|(output_dir: ).*|\1data/output/badminton_$VALUE|" | \
																											    sed -E "s/(^epochs: ).*/\1$EPOCHS_VALUE/" > "$CONFIG_FILE.tmp"
																									        
																									        # Move the temporary config back to the original config file
																										    mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
																										        
																										        # Build the Docker image with the modified config
																											    docker build -t "${IMAGE_NAME}_num_queries_${VALUE}" .
																											        
																											        # Restore the original config file
																												    mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
																											    done
