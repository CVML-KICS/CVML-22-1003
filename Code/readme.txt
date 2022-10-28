install requirement.txt by using:
pip install -r requirement.txt

add following command to run the script 
python track.py --user username --password password --host host_ip --database your_database --table your_table_name --VM_ID your_VM_ID

following is the schema of the table.
video_id PK
created_at
processing_at 
video_vm_id
input_path
stats_start_time 
stats_end_time
stats_image_size 
stats_objects_count
stats_objects_detection
stats_fps         
stats_yolo_sort_time 
stats_detection_time

