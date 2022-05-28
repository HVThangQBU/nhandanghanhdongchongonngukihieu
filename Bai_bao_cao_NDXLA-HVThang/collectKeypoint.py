import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from keypoint import mp_holistic,mediapipe_detection,draw_styled_landmarks,extract_keypoints

# def DataPath():
#     # Path for exported data, numpy arrays
#     global DATA_PATH
#     DATA_PATH = os.path.join('MP_Data') 
   
#     # Actions that we try to detect
#     global actions
#     actions = np.array(['ok', 'i love you','Good'])

#     # Thirty videos worth of data
#     global no_sequences,sequence_length
#     no_sequences = 30

#     # Videos are going to be 30 frames in length
#     sequence_length = 30
#     for action in actions: 
#         for sequence in range(no_sequences):
#             try: 
#                 os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#             except:
#                 pass
DATA_PATH = os.path.join('MP_Data') 
   
    # Actions that we try to detect
global actions
actions = np.array(['ok', 'i love you','Good','Hi'])

    # Thirty videos worth of data
global no_sequences,sequence_length
no_sequences = 30

    # Videos are going to be 30 frames in length
sequence_length = 10
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
        
def pause():
    programPause = input("Press the <ENTER> key to continue...")
def runCollect():
#    DataPath()
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            pause()
            cv2.waitKey(2000)
            for sequence in range(no_sequences):
                print('sequence',sequence)
                # Loop through video length aka sequence length
                #for frame_num in range(sequence_length):
               
                frame_num = -1
                while frame_num < sequence_length:
                    
                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
    #                 print(results)
                  
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    print('solan',frame_num)
                    
                    # NEW Apply wait logic
                    if frame_num == -1: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen

                        cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        cv2.imwrite(npy_path+'.jpg',image)
                        np.save(npy_path, keypoints)
                    frame_num +=1
                            # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    cv2.imshow('OpenCV Feed', image)
                        
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    runCollect()