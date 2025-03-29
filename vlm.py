import base64
import json
import cv2
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
from openai import OpenAI, AzureOpenAI
from VLM_CaP.src.key import projectkey
import os
import re
import argparse
import csv
import ffmpy
import ast
from VLM_CaP.src.vlm_video import extract_frame_list, base64_to_cv2  # import extract_frames
from keypoint_prompt import find_keypoint_coords
from PIL import Image
from scaffold import dot_matrix_two_dimensional_with_box, annotate_image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
task_name = ""
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, obj_name, point_coords=None, box_coords=None, input_labels=None, borders=True):
    print(" drawing masks ---------\n")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"drawing mask {i+1} ---------\n")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # plt.show()
    # save the plot
        plt.savefig(f"visualization_new/{task_name}/{obj_name}_mask_{i}.jpg")
def build_sam2_model():
    print("building sam2 model ---------\n")
    sam2_checkpoint = "../SeeDo/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
    print("sam2 model built ---------\n")
    return predictor

# set up your openai api key
# client = OpenAI(api_key=projectkey)

# gpt4v
# client = AzureOpenAI(
#         api_key=os.environ.get('OPENAI_API_KEY', projectkey),
#         azure_endpoint="https://xuhuazhe-gptv2.openai.azure.com/openai/deployments/gpt-xgw/chat/completions?api-version=2024-02-15-preview",
#         api_version="2024-02-15-preview",
#     )

# gpt4o
client = AzureOpenAI(
        api_key=os.environ.get('AZURE_OPENAI_API_KEY', projectkey),
        azure_endpoint="https://ai-xuhuazhe6145ai854228526556.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",
        api_version="2025-01-01-preview",
    )

# def for calling openai api with different prompts
def call_openai_api(prompt_messages):
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 400,
        "temperature": 0
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content
                # "Notice that there might be similar objects. You are supposed to use the index annotated on the objects to distinguish between only similar objects that is hard to distinguish with language."
def get_object_list(selected_frames):
    # first prompt to get objects in the environment
    prompt_messages_state = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a visual object detector. Your task is to count and identify the objects in the provided image. Focus on objects classified as grasped_objects and containers. "}
            ],
        },
        {
            "role": "user",
            "content": [
                # {"type": "text", "text": "There are two kinds of objects, grasped_objects and containers in the environment. Do not count in hand or person as objects."},
                {"type": "text", "text": "You will be presented wit two image frames, the first one showing the environment state when the manipulation begins, and the second one when the manipulation ends."},
                {"type": "text", "text": "Please focus on the objects or object parts that have moved or changed in state between these two frames. Other irrelevant objects can be ignored. Do not count in hand or person as objects."},
                {"type": "text", "text": "Note that relevant objects are likely in the foreground, but they can also be in the background if they are interacted with. A relevant object can a part of a larger object."},
                {"type": "text", "text": "Based on the input pictures, answer:"},
                {"type": "text", "text": "1. How many objects are there in the environment involved in the manipulation?"},
                {"type": "text", "text": "2. What are these objects?"},
                {"type": "text", "text": "You should respond in the format of the following example:"},
                {"type": "text", "text": "Number: 3"},
                {"type": "text", "text": "Objects: purple eggplant, white bowl, drawer"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{selected_frames[0]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{selected_frames[-1]}"
                    }
                }
            ],
        },
    ]
    response_state = call_openai_api(prompt_messages_state)
    return response_state
def extract_num_object(response_state):
    # extract number of objects
    num_match = re.search(r"Number: (\d+)", response_state)
    num = int(num_match.group(1)) if num_match else 0

    # extract objects
    objects_match = re.search(r"Objects: (.+)", response_state)
    print(objects_match,end=" objects_match\n")
    objects_list = objects_match.group(1).split(", ") if objects_match else []
    # 移除对象名称中的括号部分
    objects_list = [re.sub(r'\s*\([^)]*\)', '', obj) for obj in objects_list]
    print(objects_list,end=" objects_list\n")
    # construct object list
    objects = [obj.strip() for obj in objects_list]

    return num, objects
def extract_keywords_pick(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting pick keyword from response:", response)
        return None
def extract_keywords_drop(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting drop keyword from response:", response)
        return None
def extract_keywords_reference(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting reference object from response:", response)
        return None
def is_frame_relevant(response):
    return "hand is manipulating an object" in response.lower()
def parse_closest_object_and_relationship(response):
    pattern = r"Closest Object: ([^,]+), (.+)"
    match = re.search(pattern, response)
    if match:
        return match.group(1), match.group(2)
    print("Error parsing reference object and relationship from response:", response)
    return None, None

def process_images(selected_frames, obj_list, interim_frames=None, output_dir=None):
    string_cache = ""  # cache for CaP operations
    i = 0
    # clear the response_analysis.txt
    with open(os.path.join(output_dir, "response_analysis.txt"), "w") as f:
        f.truncate(0)
    sam2_predictor = build_sam2_model()
    while i < len(selected_frames) - 1:
        # input_frame_pick = selected_frames[i:i+1]
        # prompt_messages_relevance_pick = [
        #     {
        #     "role": "system",
        #     "content": [
        #         {"type": "text", "text": "You are an operations inspector. You need to check whether the hand in operation is holding an object. The objects have been outlined with contours of different colors and labeled with indexes for easier distinction."}
        #     ],
        #     },
        #     {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "This is a picture from a pick-and-drop task. Please determine if the hand is manipulating an object."},
        #         {"type": "text", "text": "Respond with 'Hand is manipulating an object' or 'Hand is not manipulating an object'."},
        #         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_frame_pick[0]}"}} if input_frame_pick else {}
        #     ],
        #     },
        # ]
        # response_relevance_pick = call_openai_api(prompt_messages_relevance_pick)
        # print(response_relevance_pick)
        # if not is_frame_relevant(response_relevance_pick):
        #     i += 1
        #     continue
        input_frame_analysis_1 = selected_frames[i]
        input_frame_analysis_2 = selected_frames[i+1]
        # interim_frame_analysis = interim_frames[i] if interim_frames else None
        last_step_description = "N/A"
        prompt_messages_analysis = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an operations inspector. You need to report what manipulation process has happened between the two image frames provided."}
                    ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze these two consecutive frames and describe the precise interaction between the hand and objects. Focus on:"},
                    {"type": "text", "text": "1. Relative positions between objects (above/below, left/right, in front of/behind, closer to/farther from)"},
                    {"type": "text", "text": "2. Hand posture and grip changes (tightening/loosening, pushing/pulling, rotating, etc.)"},
                    {"type": "text", "text": "3. Object state and movement (being moved, flipped, opened/closed, etc.)"},
                    {"type": "text", "text": "Provide your response in the following structured format:"},
                    {"type": "text", "text": "Objects: [List all visible objects in the scene]"},
                    {"type": "text", "text": "Contact Relations: [Describe which objects are touching/contacting each other]"},
                    {"type": "text", "text": "Interaction: [Describe the spatial relationship change between objects using precise action verbs and positional language]"},

                    {"type": "text", "text": "Example input:"},
                    {"type": "text", "text": "[Frame 1] Hand is hovering above a table surface. A coffee mug is on the left side of the table and a pen is on the right side. The mug and pen are not touching. Hand is not in contact with any objects."},
                    {"type": "text", "text": "[Frame 2] Hand is gripping the mug handle. The mug has been moved to the right and is now touching the pen. Both objects remain on the table surface."},

                    {"type": "text", "text": "Example output:"},
                    {"type": "text", "text": "Objects: hand, mug, pen, table"},
                    {"type": "text", "text": "Contact Relations: Hand is gripping the mug handle. The mug is touching the pen. Both mug and pen are in contact with the table."},
                    {"type": "text", "text": "Interaction: The mug is being moved closer to the pen until they touch."},

                    {"type": "text", "text": "Focus on capturing:"},
                    {"type": "text", "text": "1. All relevant objects in the scene"},
                    {"type": "text", "text": "2. All contact points between objects and hands"},
                    {"type": "text", "text": "3. Precise spatial relationship changes"},
                    {"type": "text", "text": "4. State changes of objects"},

                    {"type": "text", "text": "Provide only the structured output without additional explanation."},
                    # {"type": "text", "text": "You should respond in the format of the following examples:"},
                    # {"type": "text", "text": "Example:"},
                    # {"type": "text", "text": "Hand is open, positioned about 10cm above the table surface. On the table, a coffee mug is on the left and a pen is on the right, approximately 20cm apart. A drawer is partially open."},
                    # {"type": "text", "text": "Hand is gripping the mug handle, moving the mug about 10cm to the right, bringing the coffee mug closer to the pen. The drawer has been pushed to a fully closed position."},

                    # {"type": "text", "text": "These are two images from a manipuation task, showing two consecutive keyframes. Please analyze the manipulation process that has happened between these two frames."},
                    # {"type": "text", "text": "You need to clearly point out the names of the objects involved in the manipulation process, their motion, and their interaction with each other or with the manipulator."},
                    # {"type": "text", "text": "Keep in mind that the action between the two keyframes should be a simple process, and only the objects being interacted with in the two frames should be mentioned."},
                    # {"type": "text", "text": "Objects: mug, pen. Interaction: The mug is being moved closer to the pen."},
                    # {"type": "text", "text": "Objects: red chili. Interaction: The red chili is being picked up by the hand."},
                    {"type": "text", "text": "Note that if you consider the two images to be too similar, or there is no progress in the manipulation task, you should respond with 'No significant change'."},
                    {"type": "text", "text": "Likewise, if the two images are are completely different, for example, the scene has changed, you should respond with 'Not consecutive keyframes'."},
                    # {"type": "text", "text": f"Finally, the description you provided on the previous step is '{last_step_description}'. The 'Objects' in the last step correspond to the objects in the first image frame you see."},
                    # {"type": "text", "text": "Please identify each object name in the previous step's description with objects in the first frame, and use the same names for the same objects in your response. "},
                    {"type": "text", "text": f"If possible, please refer to the objects with names in the object list: {obj_list}."},
                    {"type": "text", "text": "Please precisely describe ONLY the action of the hand between the two keyframes, which should be independent of the previous step's description or the implied intent."},
                    {"type": "text", "text": "Describe what action is occurring between these frames using precise spatial and semantic language."},
                    # {"type": "text", "text": "If the last step's description is N/A or the interacted object is new, you can assign a fixed name to it in the 'Objects' and 'Interaction' part of your response."},
                    # {"type": "text", "text": "Additionally, if you think the previous description also accurately describes the current step, respond with 'Same as previous' instead."},

                    # First image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{input_frame_analysis_1}",
                        }
                    },
                    # Second image
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{input_frame_analysis_2}",
                        }
                    }
                    # Add more images as needed
                ]
            }
        ]
        response_analysis = call_openai_api(prompt_messages_analysis)
        print(response_analysis)
        # 从response中提取objects
        stage_object_list = []
        if "Objects:" in response_analysis and "Contact Relations:" in response_analysis:
            # 提取Objects和Interaction之间的文本
            objects_text = response_analysis.split("Contact Relations:")[0].split("Objects:")[1].strip()
            # 分割成单独的object并去除空格
            stage_object_list = [obj.strip() for obj in objects_text.split(",")]
            print("Extracted objects:", stage_object_list)
            ## merge the object list with the previous stage
            obj_list = list(set(obj_list + stage_object_list))
            print("Merged objects:", obj_list)

        ### save the response_analysis to a file
        with open(os.path.join(output_dir, "response_analysis.txt"), "a") as f:
            f.write(f"from frame {i} to frame {i+1}\n")
            f.write(response_analysis + '\n' + '------------------------------------' + '\n')


        string_cache += response_analysis + '\n' + '[SEG]' + '\n'
        last_step_description = response_analysis

        # input_frame_analysis_1_rgb = base64_to_cv2(input_frame_analysis_1)
        # object_list = response_analysis.split("Interaction:")[0].split("Objects:")[1].strip().split(',') if "Interaction:" in response_analysis else ""
        # for object_name in object_list:
        #     find_keypoint_coords(input_frame_analysis_1_rgb, object_name.strip('. '), save_path=f"./visualization/{i}_{object_name.strip('. ')}.jpg")

        prompt_message_object = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a object parser. You need to single out objects from a short description of their interaction"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The langauge description of a manipulation step is {response_analysis}. Answer:"},
                    {"type": "text", "text": f"1. Which object is being picked up?"},
                    {"type": "text", "text": f"2. Which object is being dropped?"},
                    {"type": "text", "text": f"3. What is the reference object for the placement location of the picked object?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_frame_analysis_1}"}} if input_frame_analysis_1 else {}
                ]
            }
        ]

        # prompt_message_constraint = [
        #     {
        #         "role": "system",
        #         "content": [
        #             {"type": "text", "text": "You are a constraint inspector. You need to report the spatial constraints on keypoints that are satisfied along a few image frames of object manipulation."}
        #         ],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": f"The langauge description of a manipulation step is {response_analysis}. Based on the following image frames of the manipulation process, answer:"},
        #             {"type": "text", "text": f"1. Using a natural description, what are the spatial constraints that are satisfied in the current image frame that are key to completing the step correctly?"},
        #             {"type": "text", "text": f"2. For each of these constraints, what are the keypoints that can be used to represent them? You can describe them as points on the objects."},
        #             {"type": "text", "text": f"3. If these keypoint constraints are not satisfied, what are the possible consequences?"},
        #             {"type": "text", "text": f"You should respond in the format of the following example:"},
        #             {"type": "text", "text": f"Constraint: The object being picked up should be in contact with the hand. Keypoints: The hand's fingertips and the object's surface. Consequence: If the object is not in contact with the hand, it may not be picked up successfully."},
        #             {"type": "text", "text": f"You only need to the most constraint relevant to the object currently manipulated. Other objects that are not being interacted with or static can be ignored."},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_frame_analysis_1}"}} if input_frame_analysis_1 else {},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{interim_frame_analysis}"}} if interim_frame_analysis else {},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_frame_analysis_2}"}} if input_frame_analysis_2 else {}
        #         ]
        #     }
        # ]

        # response_constraint = call_openai_api(prompt_message_constraint)
        # print(response_constraint)

        # # which to pick
        # prompt_messages_pick = [
        #     {
        #         "role": "system",
        #         "content": [
        #             "You are an operation inspector. You need to check which object is being picked in a pick-and-drop task. Some of the objects have been outlined with contours of different colors and labeled with indexes for easier distinction.",
        #             "The contour and index is only used to help. Due to limitation of vision models, the contours and index labels might not cover every objects in the environment. If you notice any unannotated objects in the demo or in the object list, make sure you name it and handle them properly.",
        #         ],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             f"This is a picture describing the pick state of a pick-and-drop task. The objects in the environment are {obj_list}. One of the objects is being picked by a human hand or robot gripper now. The objects have been outlined with contours of different colors and labeled with indexes for easier distinction.",
        #             "Based on the input picture and object list, answer:",
        #             "1. Which object is being picked",
        #             "You should respond in the format of the following example:",
        #             "Object Picked: red block",
        #             *map(lambda x: {"image": x, "resize": 768}, input_frame_pick),
        #         ],
        #     },
        # ]
        # response_pick = call_openai_api(prompt_messages_pick)
        # print(response_pick)
        # object_picked = extract_keywords_pick(response_pick)
        # i += 1
        # # Ensure there is another frame for drop and relative position reasoning
        # if i >= len(selected_frames):
        #     break
        # # Check if the second frame (i) is relevant (i.e., hand is holding an object)
        # input_frame_drop = selected_frames[i:i+1]
        # # reference object
        # prompt_messages_reference = [
        #     {
        #         "role": "system",
        #         "content": [
        #             "You are an operation inspector. You need to find the reference object for the placement location of the picked object in the pick-and-place process. Notice that the reference object can vary based on the task. If this is a storage task, the reference object should be the container into which the items are stored. If this is a stacking task, the reference object should be the object that best expresses the orientation of the arrangement."
        #         ],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             f"This is a picture describing the drop state of a pick-and-place task. The objects in the environment are {obj_list}. {object_picked} is being dropped by a human hand or robot gripper now.",
        #             "Based on the input picture and object list, answer:",
        #             f"1. Which object in the rest of object list do you choose as a reference object to {object_picked}",
        #             "You should respond in the format of the following example without any additional information or reason steps:",
        #             "Reference Object: red block",
        #             *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
        #         ],
        #     },
        # ]
        # response_reference = call_openai_api(prompt_messages_reference)
        # print(response_reference)
        # object_reference = extract_keywords_reference(response_reference)
        # # current_bbx = bbx_list[i] if i < len(bbx_list) else {}

        #             # "Due to limitation of vision models, the contours and index labels might not cover every objects in the environment. If you notice any unannotated objects in the demo or in the object list, make sure you handle them properly.",
        # prompt_messages_relationship = [
        #     {
        #         "role": "system",
        #         "content": [
        #             "You are a VLMTutor. You will describe the drop state of a pick-and-drop task from a demo picture. You must pay specific attention to the spatial relationship between picked object and reference object in the picture and be correct and accurate with directions.",
        #         ],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             f"This is a picture describing the drop state of a pick-and-drop task. The objects in the environment are object list: {obj_list}. {object_picked} is said to be being dropped by a human hand or robot gripper now.",
        #             f"However, the object being dropped might be wrong due to bad visual prompt. If you feel that object being picked is not {object_picked} but some other object, red chili is said to be the object picked but you feel it is an orange carrot, you MUST modify it and change the name!"
        #             # "But notice that due to limitation of vision models, the contours and index labels might not cover every objects in the environment. If you notice any unannotated objects in the demo or in the object list, make sure you mention their name and handle their spatial relationships."
        #             # "The ID is only used to help with your reasoning. You should only mention them when the objects are the same in language description. For example, when there are two white bowls, you must specify white bowl (ID:1), white bowl (ID:2) in your answer. But for different objects like vegetables, you do not need to specify their IDs."
        #             # f"To help you better understand the spatial relationship, a bounding box list is given to you. Notice that the bounding boxes of objects in the bounding box list are distinguished by labels. These labels correspond one-to-one with the labels of the objects in the image. The bounding box list is: {bbx_list}",
        #             # "The coordinates of the bounding box represent the center point of the object. The format is two coordinates (x,y). The origin of the coordinates is at the top-left corner of the image. If there are two objects A(x1, y1) and B(x2, y2), a significantly smaller x2 compared to x1 indicates that B is to the left of A; a significantly greater x2 compared to x1 indicates that B is to the right of A; a significantly smaller y2 compared to y1 indicates that B is at the back of A;  a significantly greater y2 compared to y1 indicates that B is in front of A."
        #             # "Pay attention to distinguish between at the back of and on top of. If B and A has a visual gap, they are not in touch. Thus B is at the back of A. However, if they are very close, this means B and A are in contact, thus B is on top of A."
        #             # "Notice that the largest difference in corresponding coordinates often represents the most significant feature. If you have coordinates with small difference in x but large difference in y, then coordinates y will represent most significant feature. Make sure to use the picture together with coordinates."
        #             f"The object picked is being dropped somewhere near {object_reference}. Based on the input picture, object list answer:",
        #             f"Drop object picked to which relative position to the {object_reference}? You need to mention the name of objects in your answer.",
        #             f"There are totally six kinds of relative position, and the direction means the visual direction of the picture.",
        #             f"1. In (object picked is contained in the {object_reference})",
        #             f"2. On top of (object picked is stacked on the {object_reference}, {object_reference} supports object picked)",
        #             f"3. At the back of (in demo it means object picked is positioned farther to the viewer relative to the {object_reference})",
        #             f"4. In front of (in demo it means object picked is positioned closer to the viewer or relative to the {object_reference})",
        #             "5. to the left",
        #             "6. to the right",
        #             f"You must choose one relative position."
        #             "You should respond in the format of the following example without any additional information or reason steps, be sure to mention the object picked and reference object.",
        #             f"Drop yellow corn to the left of the red chili",
        #             f"Drop red chili in the white bowl",
        #             f"Drop wooden block (ID:1) to the right of the wooden block (ID:0)",
        #             *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
        #         ],
        #     },
        # ]
        # response_relationship = call_openai_api(prompt_messages_relationship)
        # print(response_relationship)
        # string_cache += response_relationship + " and then "

        i += 1

    prompt_image_list = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                        }
                    } for frame in selected_frames]

    prompt_messages_check = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a descroption checker. You need to check and improve descriptions for a manipulation task plan based on image frames."}
                    ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You will be provided with a text description of a sequence of actions that constitutes a manipulation task. The description is based on two consecutive keyframes from the task. Please analyze the description and check if it accurately describes the manipulation process that has happened between these two frames."},
                    {"type": "text", "text": "Keep in mind that each segment of description composed of a 'Objects' and 'Interaction' part corresponds to the interval between two consecutive keyframes. That is, the first description should describe the action between the first and second keyframes, the second description should describe the action between the second and third keyframes, and so on."},
                    {"type": "text", "text": "You need to pay attention to the names of the objects mentioned, and the motion and interaction of the objects with each other or with the manipulator."},
                    {"type": "text", "text": "If an object is referred to by a different name at some step than other steps, you should correct it. If necessary, correct the actions so that the actions form a coherent sequence that complete a manipulation task."},
                    {"type": "text", "text": "You can only correct the descriptions within each step. Do not create new steps or delete old ones."},
                    {"type": "text", "text": "Please retain the format of the response as 'Objects: object1, object2, ... Interaction: action description. [SEG] ... ' Directly respond with the revised description without any additional information or explanation."},
                    {"type": "text", "text": f"The dscription is {string_cache}. The image frames are as the following:"},
                    # {"type": "text", "text": "If the last step's description is N/A or the interacted object is new, you can assign a fixed name to it in the 'Objects' and 'Interaction' part of your response."},
                    # {"type": "text", "text": "Additionally, if you think the previous description also accurately describes the current step, respond with 'Same as previous' instead."},
                ] + prompt_image_list
            }
        ]

    response_analysis_checked = call_openai_api(prompt_messages_check)
    # print("Revised response:", response_analysis_checked)
    # string_cache += response_analysis + '\n'
    # last_step_description = response_analysis

    revised_response_list = response_analysis_checked.strip('\n ').split('[SEG]')
    revised_response_list = [response.strip('\n ') for response in revised_response_list if 'Objects:' in response and 'Interaction:' in response]
    print("Revised response list:", revised_response_list)

    # save the revised response list to the same file
    with open(os.path.join(output_dir, "response_analysis.txt"), "a") as f:
        f.write("Revised response list:\n")
        f.write(response_analysis_checked)


    constraint_desc_list = []
    obj_keypoint_dict = dict()
    constraint_obj_list = obj_list.copy()
    obj_scaffold_list = dict()
    scaffold_grid_dict = dict()
    start_image = None
    detected_object_list = []


    constraint_dict_list = []
    for i in range(len(revised_response_list)):
        input_frame_analysis_1 = selected_frames[i]
        input_frame_analysis_2 = selected_frames[i+1]
        interim_frame_analysis = interim_frames[i] if interim_frames else None

        frame_analysis = revised_response_list[i]

        if i==0:
            input_frame_analysis_1_rgb = base64_to_cv2(input_frame_analysis_1)
            start_image_array = input_frame_analysis_1_rgb
            start_image_array = cv2.cvtColor(input_frame_analysis_1_rgb, cv2.COLOR_BGR2RGB)
            start_image = Image.fromarray(start_image_array)
            ## save start_image
            # import pdb; pdb.set_trace()
            start_image.save(f"visualization_new/{task_name}/start_image_{i}.jpg")
            w, h = start_image.size
            frame_object_list = frame_analysis.split("Interaction:")[0].split("Objects:")[1].strip().split(',') if "Interaction:" in response_analysis else ""
            for object_name in obj_list:
                start_image_array_copy = start_image_array.copy()
                start_image_copy = start_image.copy()
                # use start_image_array_copy to find keypoint coords
                box_coords = find_keypoint_coords(start_image_array_copy, object_name.strip('. '), save_path=f"visualization_new/{task_name}/{i}_{object_name.strip('. ')}.jpg")


                # print(box_coords,end=f'  box coords of {object_name/}\n')
                if box_coords is not None:
                    detected_object_list.append(object_name)
                else:
                    continue

                box_coords  = box_coords[0] * np.array([w, h, w, h])
                # box_center = int((box_coords[0] + box_coords[2]) / 2), int((box_coords[1] + box_coords[3]) / 2)
                # box_width = 2 * np.ceil(max(box_coords[0] - box_center[0], box_center[0] - box_coords[2]))
                # box_height = 2 * np.ceil(max(box_coords[1] - box_center[1], box_center[1] - box_coords[3]))
                box_center = round(box_coords[0]), round(box_coords[1])
                box_width = round(box_coords[2])
                box_height = round(box_coords[3])
                # import pdb; pdb.set_trace()
                ### save pic of start_image_copy with box
                ## 不用 imagedraw 画框，而是用 cv2 画框
                box_left = box_center[0] - box_width // 2
                box_top = box_center[1] - box_height // 2
                box_right = box_center[0] + box_width // 2
                box_bottom = box_center[1] + box_height // 2
                box = np.array([box_left, box_top, box_right, box_bottom])
                # cv2.rectangle(start_image_copy, (box_left, box_top), (box_left + box_width, box_top + box_height), (0, 0, 255), 2)
                # start_image_copy.save(f"visualization_new/{task_name}/start_image_copy_with_box_{object_name.strip('. ')}.jpg")
                sam2_predictor.set_image(start_image_copy)
                masks, scores, _ = sam2_predictor.predict(box=[box_left, box_top, box_right, box_bottom])
                print(masks,end=f'  masks of {object_name}\n')
                print(scores,end=f'  scores of {object_name}\n')

                show_masks(start_image_copy, masks, scores, object_name, box_coords=box)
                # import pdb; pdb.set_trace()
                # scaffold_img, (box_left, box_top), (cell_width, cell_height) = dot_matrix_two_dimensional_with_box(
                #     image_or_image_path=start_image_copy,
                #     save_path=f"visualization_new/{task_name}/scaffold_{object_name.strip('. ')}.jpg",
                #     save_img=True,
                #     box_width=box_width,
                #     box_height=box_height,
                #     box_coords=box_center,
                #     font_path="/usr/share/fonts/truetype/arial/arial.ttf"
                # )
                # highest score
                highest_score_index = np.argmax(scores)
                mask = masks[highest_score_index]
                scaffold_img, sampled_coords = annotate_image(start_image_copy, save_path=f"visualization_new/{task_name}/annotated_scaffold_{object_name.strip('. ')}.jpg", mask=mask)
                obj_scaffold_list[object_name.strip('. ')] = scaffold_img
                scaffold_grid_dict[object_name.strip('. ')] = {
                    "box_left": box_left,
                    "box_top": box_top,
                    "cell_width": box_width,
                    "cell_height": box_height,
                    "sampled_coords": sampled_coords
                }
            print(detected_object_list,end=f'  detected object list\n')



        example_json = {
            "Subpath": {
                "Constraint": "The cube being picked up should be in contact with the hand.",
                "Keypoints": {
                    "hand": "the hand's fingertips",
                    "cube": "the cube's surface",
                },
            },
            "Subgoal": {
                "Constraint": "The cube should be placed above the plate.",
                "Keypoints": {
                    "plate": "the center of the plate",
                    "cube": "the center of the cube",
                },

            },
        }

        example_json_2 = {
            "Subpath": 'N/A',
            "Subgoal": {
                "Constraint": "The apple should be inside the basket.",
                "Keypoints": {
                    "apple": "the center of the apple",
                    "basket": "the center of the opening of the basket",
                },
            },
        }

        json_string = json.dumps(example_json, indent=4)

        json_string_2 = json.dumps(example_json_2, indent=4)

        # convert back to json
        # prompt_message_constraint = json.loads(json_string)

        prompt_message_constraint = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a constraint inspector. You need to report the spatial constraints on keypoints that are satisfied along a few image frames of object manipulation."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The langauge description of a manipulation step is {frame_analysis}. Based on the following image frames of the manipulation process, answer:"},
                    {"type": "text", "text": f"1. Using a natural description, what are the spatial constraints that are satisfied in the current image frame that are key to completing the step correctly?"},
                    {"type": "text", "text": f"2. For each of these constraints, what are the keypoints that can be used to represent them? You can describe them as points on the objects."},
                    {"type": "text", "text": f"Pay attention to the constraints satisfied during this manipulation process, and the ones satisfied at the final frame."},
                    {"type": "text", "text": f"In general, you should provide them as a 'Subpath' constraint and a 'Subgoal' constraint, respectively."},
                    {"type": "text", "text": f"However, if the 'Subpath' or 'Subgoal' constraint is not clear, you can respond 'N/A' in place of 'Constraint' and 'Keypoints'."},
                    # {"type": "text", "text": f"3. If these keypoint constraints are not satisfied, what are the possible consequences?"},
                    {"type": "text", "text": f"You should respond in the string format of the following examples: {json_string}. "},
                    {"type": "text", "text": f"Another example: {json_string_2}."},
                    {"type": "text", "text": f"One should be able to convert your response directly to a dicitonary. Do not add any additional comments."},
                    # {"type": "text", "text": f"Constraint: The object being picked up should be in contact with the hand. Keypoints: The hand's fingertips and the object's surface. Consequence: If the object is not in contact with the hand, it may not be picked up successfully."},
                    {"type": "text", "text": f"Please use the same names for the objects in your response as in the object list: {frame_object_list}."},
                    {"type": "text", "text": f"You only need to return the constraints most relevant to the objects currently manipulated, interacted with, or associated with the goal."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_frame_analysis_1}"}} if input_frame_analysis_1 else {},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{interim_frame_analysis}"}} if interim_frame_analysis else {},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_frame_analysis_2}"}} if input_frame_analysis_2 else {}
                ]
            }
        ]

        response_constraint = call_openai_api(prompt_message_constraint)
        try:
            constraint_dict = json.loads(response_constraint)
        except json.JSONDecodeError:
            constraint_str = response_constraint.strip("```").strip('json')
            constraint_dict = json.loads(constraint_str)
        print("Constraint response:", constraint_dict)

        constraint_dict_list.append(constraint_dict)

        if isinstance(constraint_dict['Subpath'], dict):
            subpath_constraint_dict = constraint_dict["Subpath"]
            if subpath_constraint_dict != 'N/A':
                constraint_desc_list.append(subpath_constraint_dict["Constraint"])
                for obj_name, keypoint in subpath_constraint_dict["Keypoints"].items():
                    if obj_name not in obj_keypoint_dict:
                        obj_keypoint_dict[obj_name] = []
                    obj_keypoint_dict[obj_name].append(keypoint)

        if isinstance(constraint_dict['Subgoal'], dict):
            subgoal_constraint_dict = constraint_dict["Subgoal"]
            if subgoal_constraint_dict != 'N/A':
                constraint_desc_list.append(subgoal_constraint_dict["Constraint"])
                for obj_name, keypoint in subgoal_constraint_dict["Keypoints"].items():
                    if obj_name not in obj_keypoint_dict:
                        obj_keypoint_dict[obj_name] = []
                    obj_keypoint_dict[obj_name].append(keypoint)

    keypoint_coord_list = select_keypoints(start_image, obj_scaffold_list, obj_keypoint_dict, constraint_obj_list, scaffold_grid_dict)

    # save start_image
    os.makedirs(f"./rekep_ready/{task_name}", exist_ok=True)
    start_image.save(f"./rekep_ready/{task_name}/start_image.jpg")
    start_image_path = os.path.abspath(f"./rekep_ready/{task_name}/start_image.jpg")
    print(start_image_path)

    # organize the substage descriptions and constraint descriptions
    constraint_compendium = ""
    plan_string = ""
    for step_idx in range(len(revised_response_list)):
        step_desc = revised_response_list[step_idx]
        constraint_desc = constraint_dict_list[step_idx]
        interaction_desc = step_desc.split("Interaction:")[1].strip('\n ')
        plan_string += f"{interaction_desc}\n"
        path_constraint_desc = constraint_desc["Subpath"]["Constraint"].strip('\n ')
        subgoal_constraint_desc = constraint_desc["Subgoal"]["Constraint"].strip('\n ')
        constraint_compendium += f"Description: {interaction_desc}\n"
        constraint_compendium += f"Path Constraint: {path_constraint_desc}\n"
        constraint_compendium += f"Subgoal Constraint: {subgoal_constraint_desc}\n \n"

    composed_json_dict = {
        "Plan": plan_string,
        "Constraints": constraint_compendium,
        "Keypoints": keypoint_coord_list,
        "Keypoint_Image_Path": start_image_path,
    }
     # save as json
    composed_json_string = json.dumps(composed_json_dict, indent=4)
    output_file_path = os.path.join(os.path.dirname(start_image_path), f"{task_name}_output.json")
    with open(output_file_path, "w") as json_file:
        json_file.write(composed_json_string)

    return constraint_compendium, plan_string, keypoint_coord_list

def select_keypoints(start_image, scaffold_img_dict, keypoint_dict, obj_list, scaffold_grid_dict):
    all_keypoints_list = []
    for obj_name, keypoints in keypoint_dict.items():
        print(obj_name,end=f'  obj name\n')
        # delete space in beginning and end of obj_name
        obj_name = obj_name.strip()
        print(keypoints,end=f'  keypoints\n')
        if obj_name in obj_list:
            print(obj_name,end=f'  obj name\n')
            print(keypoints,end=f'  keypoints\n')
            scaffold_params = scaffold_grid_dict.get(obj_name, None)
            scaffold_img = scaffold_img_dict.get(obj_name, None)
            if scaffold_img is None:
                print(f"Scaffold image for {obj_name} not found.")
                continue
            # assert scaffold_img is not None, f"Scaffold image for {obj_name} not found."
            # Convert scaffold_img to JPEG format
            scaffold_img_cv2 = cv2.cvtColor(np.array(scaffold_img), cv2.COLOR_RGB2BGR)
            _, scaffold_img_encoded = cv2.imencode('.jpg', scaffold_img_cv2)
            scaffold_img_base64 = base64.b64encode(scaffold_img_encoded).decode('utf-8')
            # prompt_message_keypoint = [
            #     {
            #         "role": "system",
            #         "content": [
            #             {"type": "text", "text": "You are a keypoint selector. You need to select a grid point on an image that best fits the description."}
            #         ],
            #     },
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": f"You are provided with an image in which one onject is overlaid with a grid of points. For each of the following keypoint decsriptions, please select the point that best fits the description of the keypoint on the object."},
            #             {"type": "text", "text": f"If the description refers to a small part or a point on the object, select the grid point that is closest to the point. If the description refers to a large part or a region on the object, select the grid point that is closest to the center of the part."},
            #             {"type": "text", "text": f"Each candidate grid point is marked with a tuple of integers slighty to the right and below the point. The coordinates are in the format (x, y), where x is the horizontal index and y is the vertical index. "},
            #             {"type": "text", "text": f"Please report your selection in the format of a list of tuples (x,y), with no additional text or explanation. Each tuple corresponds to a provided keypoint description in the same order. You should respond in the format of the following example:"},
            #             {"type": "text", "text": f"[(1, 2), (2,3)]"},
            #             {"type": "text", "text": f"The object being described is {obj_name}. If multiple descriptions refer to the same keypoint, you may only return distinct keypoints in your response."},
            #             {"type": "text", "text": f"The keypoint descriptions are {keypoints}, and the overlaid image is as follows:"},
            #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{scaffold_img_base64}"}},
            #         ]
            #     }
            # ]
            # set a prompt to select the keypoints not for grid points but for some circled points with black edge and number inside the circle
            prompt_message_keypoint = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a keypoint selector.  I have annotated the image with numbered circles. You need to select some circled points with black edge and number inside the circle on an image that best fits the description."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Please report your selection in the format of a list of numbers with no additional text or explanation. Each number corresponds to a provided keypoint description in the same order. You should respond in the format of the following example:"},
                        {"type": "text", "text": f"[0, 8, 9]"},
                        {"type": "text", "text": f"The object being described is {obj_name}. If multiple descriptions refer to the same keypoint, you may only return distinct keypoints in your response."},
                        {"type": "text", "text": f"The keypoint descriptions are {keypoints}, and the overlaid image is as follows:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{scaffold_img_base64}"}},
                    ]
                }
            ]
            response_keypoint = call_openai_api(prompt_message_keypoint)
            keypoint_index_list = eval(response_keypoint) if response_keypoint else []
            print("Keypoint response:", keypoint_index_list)
            print("Keypoints:", keypoints)
            # Check if the response is a list of tuples

            # if not isinstance(keypoint_index_list, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in keypoint_index_list):
            #     print(f"Invalid response format for {obj_name}: {response_keypoint}")
            #     continue

            keypoint_coord_list, annotated_image_np = visualize_annotate_keypoints_on_image(start_image, keypoint_index_list, scaffold_params, obj_name)

            # if not isinstance(keypoint_index_list, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in keypoint_index_list):
            #     print(f"Invalid response format for {obj_name}: {response_keypoint}")
            #     continue
            # keypoint_coord_list, annotated_image_np = annotate_keypoints_on_image(start_image, keypoint_index_list, scaffold_params, obj_name)
            all_keypoints_list.extend(keypoint_coord_list)
            # Draw the keypoints on the image
            # for point in keypoints:
            #     x, y = point
            #     cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            # # Add a label for the object
            # cv2.putText(annotated_image, obj_name, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return all_keypoints_list

# def annotate_keypoints_on_image(start_image, keypoint_index_list, scaffold_params, obj_name):
#     ## for grid points
#     # Create a copy of the start image to draw on
#     keypoint_coord_list = []
#     annotated_image = start_image.copy()
#     annotated_image_np = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
#     # Draw keypoints and labels on the image
#     for index, point in enumerate(keypoint_index_list):
#         import pdb; pdb.set_trace()
#         print(point,end=f'  point\n')
#         y, x = point
#         # Calculate the position for the label
#         label_x = scaffold_params["box_left"] + (x-1) * scaffold_params["cell_width"]
#         label_y = scaffold_params["box_top"] + (y-1) * scaffold_params["cell_height"]
#         keypoint_coord_list.append((int(label_x), int(label_y)))
#         cv2.circle(annotated_image_np, (int(label_x), int(label_y)), 4, (0, 255, 0), -1)
#         cv2.putText(annotated_image_np, str(y) + ', ' + str(x), (int(label_x) + 10, int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         # save fig
#     cv2.imwrite(f"visualization_new/{task_name}/keypoints_{obj_name}.jpg", annotated_image_np)

#     # return annotated_image_np
#     print(f"keypoint {y},{x}, position: ", int(label_x), int(label_y))
#     # cv2.imwrite(f"visualization_new/{task_name}/keypoints_{obj_name}.jpg", annotated_image_np)

#     return keypoint_coord_list, annotated_image_np

def visualize_annotate_keypoints_on_image(start_image, keypoint_index_list, scaffold_params, obj_name):
    ## new version of annotate keypoints on image
    # Create a copy of the start image to draw on
    annotated_image = start_image.copy()
    keypoint_coord_list = []
    annotated_image_np = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    coords = scaffold_params["sampled_coords"]
    # Draw keypoints and labels on the image
    for index, id in enumerate(keypoint_index_list):
        y, x = coords[id]
        # Calculate the position for the label
        # label_x = scaffold_params["box_left"] + (x-1) * scaffold_params["cell_width"]
        # label_y = scaffold_params["box_top"] + (y-1) * scaffold_params["cell_height"]
        cv2.circle(annotated_image_np, (int(x), int(y)), 5, (0, 255, 0), -1)
        keypoint_coord_list.append((int(x), int(y)))
        cv2.putText(annotated_image_np, str(y) + ', ' + str(x), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # save fig
    cv2.imwrite(f"./visualization/visualized_keypoints_{obj_name}.jpg", annotated_image_np)

    return keypoint_coord_list, annotated_image_np


def save_results_to_csv(demo_name, num, obj_list, string_cache, output_file):
    file_exists = os.path.exists(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["demo", "object", "action list"])

        writer.writerow([f"{demo_name}", f"{num} objects: {', '.join(obj_list)}", string_cache])
    print(f"Results appended to {output_file}")

def convert_video_to_mp4(input_path):
    """
    Converts the input video file to H.264 encoded .mp4 format using ffmpy.
    The output path will be the same as the input path with '_converted' appended before the extension.
    """
    # Get the file name without extension and append '_converted'
    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}_converted.mp4"
    # Run FFmpeg command to convert the video
    ff = ffmpy.FFmpeg(
        inputs={input_path: None},
        outputs={output_path: '-c:v libx264 -crf 23 -preset fast -r 30'}
    )
    ff.run()
    print(f"Video converted successfully: {output_path}")
    return output_path

def get_interim_frame_index(frame_index_list):
    # for each interval between the keyframes, select the middle frame as the interim frame
    interim_frame_index = []
    for i in range(len(frame_index_list) - 1):
        start_index = frame_index_list[i]
        end_index = frame_index_list[i + 1]
        # Calculate the middle index
        middle_index = (start_index + end_index) // 2
        interim_frame_index.append(middle_index)
    return interim_frame_index

def main(input_video_path, frame_index_list, bbx_list, output_dir):
    ### create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    video_path = input_video_path
    # list to store key frames
    selected_raw_frames1 = []
    interim_raw_frames1 = []
    # list to store key frame indexes
    frame_index_list = ast.literal_eval(frame_index_list)
    if 0 not in frame_index_list:
        frame_index_list.insert(0, 0)
    print('---SELECTED FRAMES---', frame_index_list)
    interim_index_list = get_interim_frame_index(frame_index_list)
    print('---INTERIM FRAMES---', interim_index_list)
    selected_frame_index = frame_index_list
    # Convert the video to H.264 encoded .mp4 format
    # converted_video_path = convert_video_to_mp4(video_path)
    # exit(0)
    # Open the converted video
    cap = cv2.VideoCapture(video_path)
    # Manually calculate total number of frames
    actual_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        actual_frame_count += 1
    # Reset the capture to the beginning of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Actual frame count: {actual_frame_count}")
    # Iterate through index list and get the frame list
    for index in selected_frame_index:
        if index < actual_frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, cv2_image = cap.read()
            if ret:
                selected_raw_frames1.append(cv2_image)
            else:
                print(f"Failed to retrieve frame at index {index}")
        else:
            print(f"Frame index {index} is out of range for this video.")
    for index in interim_index_list:
        if index < actual_frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, cv2_image = cap.read()
            if ret:
                interim_raw_frames1.append(cv2_image)
                # save img
                interim_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                interim_image = Image.fromarray(interim_image)
                interim_image.save(f"./visualization/interim_{index}.jpg")
            else:
                print(f"Failed to retrieve frame at index {index}")
        else:
            print(f"Frame index {index} is out of range for this video.")
    # Release video capture object
    cap.release()
    selected_frames1 = extract_frame_list(selected_raw_frames1)
    interim_frames1 = extract_frame_list(interim_raw_frames1)
    response_state = get_object_list(selected_frames1)
    num, obj_list = extract_num_object(response_state)
    print("Number of objects:", num)
    print("available objects:", obj_list)
    # obj_list = "green corn, orange carrot, red pepper, white bowl, glass container"
    # process the key frames
    # exit(0)
    string_cache = process_images(selected_frames1, obj_list, interim_frames1, output_dir)
    # if string_cache.endswith(" and then "):
    #     my_string = string_cache.removesuffix(" and then ")
    # print(string_cache)
    return string_cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and key frame extraction.")
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--list', type=str, required=True, help='List of key frame indexes')
    parser.add_argument('--bbx_list', type=str, required=True, help='Bbx of key frames')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    task_name = args.input.split("/")[-1].split(".")[0]
    print(task_name,end=" task_name\n")
    os.makedirs(f"visualization_new/{task_name}", exist_ok=True)

    # delete all files in the visualization_new folder
    folder_path = f"visualization_new/{task_name}"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    # exit(0)
    # Call the main function with arguments
    main(args.input, args.list, args.bbx_list, args.output)