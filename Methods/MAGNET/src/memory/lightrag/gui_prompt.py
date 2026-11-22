PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ATTRIBUTE_TYPES"] = ["location", "function", "shape"]

# TODO: merge_document prompts are not adapted based on the model used
# These prompts need further engineering
PROMPTS["merge_document_v1"] = """---Role---
You are a professional assistant responsible for merging similar UI elements.

---Goal---
Merge similar UI elements based on the function descriptions by semantically combining them into a single description and then updating the other attributes accordingly.

---Steps---
You will be provided two UI elements, where each element contains a **string** that describes its **function** and a **dictionary** of **attributes** (such as possible shapes and possible locations).
You should follow these steps:
1. Semantically merge the function descriptions:
    - If two UI elements share the same or similar function, combine their function descriptions into a single, coherent phrase.
    - If the functions are slightly different but related, create a new description that encompasses both.
    - Ensure the merged description is concise and easy to understand.
2. Update the attribute dictionary:
    - Combine the attributes of the two UI elements into a single dictionary. Here is how: 
        - "shape": If two elements have the same shape, keep it. If there are multiple possible shapes, check if any shapes are similar. Merge similar shapes and keep them as a list. 
        - "location": If two elements have the same location, keep it. If there are multiple possible locations, check if any locations are spatially similar. Merge similar locations and keep them as a list.
3. Return the merged UI element with the updated function description and attributes in the following form.
    {{
        "function": "merged function description",
        "attributes": {{
            "shape": ["merged shape descriptions"],
            "location": ["merged location descriptions"]
        }}
    }}

######################
---Examples---
######################
{examples}

######################
---Query---
Input:
- UI Element 1: {first_ui_element}
- UI Element 2: {second_ui_element}

######################
Output:
"""


PROMPTS["merge_document_examples_v1"] = ["""Example 1:

Input:
- UI Element 1: {"function": "click on the Google Chrome app to search", "attributes": {"shape": "a multicolored circle composed of blue, red, yellow and green segments", "location": "top-right corner of the screen"}}
- UI Element 2: {"function": "click on the Chrome app to search for the news", "attributes": {"shape": "a multicolored circle", "location": "bottom of the screen"}}

######################
Output:
{
    "function": "click on the Chrome app to search",
    "attributes": {
        "shape": ["a multicolored circle composed of blue, red, yellow and green segments"],
        "location": ["top-right corner of the screen", "bottom of the screen"]
    }
}
""", """Example 2:

Input:
- UI Element 1: {"function": "click on the shopping cart icon to add the airpods to cart", "attributes": {"shape": "a shopping cart", "location": "botom-right corner of the shopping item"}}
- UI Element 2: {"function": "click on the cart icon to view the contents in current shopping cart", "attributes": {"shape": "a shopping cart", "location": "top-right corner of the screen"}}

######################
Output:
{
    "function": "click on the shopping cart icon to add items to the cart or view its contents.",
    "attributes": {
        "shape": ["a shopping cart"],
        "location": ["botom-right corner of the shopping item", "top-right corner of the screen"]
    }
}
""", """Example 3:

Input:
- UI Element 1: {"function": "click on the close icon to close this page", "attributes": {"shape": "a cross mark", "location": "top-right corner of the screen"}}
- UI Element 2: {"function": "click on the cross mark to clear the input", "attributes": {"shape": "an X icon", "location": "right side of the search bar"}}

######################
Output:
{
    "function": "click on the close icon to close the page or clear the input.",
    "attributes": {
        "shape": ["a cross mark", "an X icon"],
        "location": ["top-right corner of the screen", "right side of the search bar"]
    }
}
"""]

PROMPTS["judge_document_v1"] = """---Role---
You are a professional assistant responsible for evaluating the consistency of UI elements.

---Goal---
Determine whether two given UI elements contain equivalent amounts of information based on their function descriptions and attributes.

---Steps---
You will be provided two UI elements, each consisting of:
- A **string** that describes its **function**.
- A **dictionary** of **attributes** (e.g., possible shapes and locations).

Follow these steps:
1. **Compare the function descriptions**:
    - If both descriptions convey the same or equivalent functionality, they are considered consistent.
    - If one description contains significantly more detail or functionality than the other, they are inconsistent.
2. **Compare the attribute dictionaries**:
    - Attributes are considered consistent if they describe the same set of properties or if minor variations do not significantly impact the conveyed information.
    - If one UI element has attributes missing or significantly different from the other, they are inconsistent.
3. **Final Judgment**:
    - If both function descriptions and attribute dictionaries contain equivalent amounts of information, return `True`.
    - Otherwise, return `False`.

######################
---Examples---
######################
{examples}

######################
---Query---
Input:
- UI Element 1: {first_ui_element}
- UI Element 2: {second_ui_element}

######################
Output:
"""  

PROMPTS["judge_document_examples_v1"] = ["""Example 1:

Input:
- UI Element 1: {"function": "click on the Google Chrome app to search", "attributes": {"shape": "a multicolored circle composed of blue, red, yellow and green segments", "location": "top-right corner of the screen"}}
- UI Element 2: {"function": "click on the Chrome app to search for the news", "attributes": {"shape": "a multicolored circle", "location": "bottom of the screen"}}

######################
Output:
True
""", """Example 2:

Input:
- UI Element 1: {"function": "click on the shopping cart icon to add the airpods to cart", "attributes": {"shape": "a shopping cart", "location": "botom-right corner of the shopping item"}}
- UI Element 2: {"function": "click on the cart icon to view the contents in current shopping cart", "attributes": {"shape": "a shopping cart", "location": "top-right corner of the screen"}}

######################
Output:
True
""", """Example 3:

Input:
- UI Element 1: {"function": "click on the close icon to close this page", "attributes": {"shape": "a cross mark", "location": "top-right corner of the screen"}}
- UI Element 2: {"function": "click on the cross mark to clear the input", "attributes": {"shape": "an X icon", "location": "right side of the search bar"}}

######################
Output:
False
"""]
