from lib2to3.fixes.fix_input import context
import json
import openai

def generate_transformation_description(api_key, json_input):
    openai.api_key = api_key
    prompt = f"""
        Let's play a game where you are transforming an input grid of numbers into an output grid of numbers.

        The numbers represent different colors:
        0 = black
        1 = blue
        2 = red
        3 = green
        4 = yellow
        5 = gray
        6 = magenta
        7 = orange
        8 = cyan
        9 = brown

        Please identify a single, generalized pattern that describes the transformation process observed across all training examples. Focus on common rules that apply to each input grid to produce the corresponding output grid, without detailing individual examples separately. Do not include the full grid data in your explanation, but focus only on the transformation steps.

        ### Training Data:
        {json_input['train']}

         ### Generalized Transformation Pattern
        Describe a single, unified set of steps that apply across all examples, summarizing the transformation logic pattern from input to output based on observed patterns.
        """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides the correct transformation logic for a given input grid. Be sure to notice that some transformations may not only have one pattern, be sure notice if there're other patterns that are combined from the training data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=1.0
    )

    # Return the generated transformation description
    return response.choices[0].message.content.strip()

def apply_transformation(api_key, description, test_input, train_input):
    openai.api_key = api_key
    # Construct the prompt with the transformation description and test input
    prompt = f"""

            ### Test Input:
            {test_input}

            ### Expected Transformation
            Only provide the output grid after applying the transformation pattern, without repeating the input grid or adding additional explanations.
            """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": f"""Based on the following transformation pattern:
            {description}
            , and based on these examples which demonstrate the pattern:
            {train_input}
            Apply this pattern to transform the given test input grid into its corresponding output grid and only output the grid."""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.9
    )

    # Return the generated output grid
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    try:
        api_key = "key"
        with open('./training/0a938d79.json') as f:
            json_input = json.load(f)
        # json_input = {
        #   "train": [
        #     {
        #       "input": [
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        #       ],
        #       "output": [
        #         [8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 3, 1, 3],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
        #         [8, 3, 1, 3, 8, 1, 8, 8, 8, 1, 3, 1, 3],
        #         [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [8, 3, 1, 3, 8, 1, 8, 8, 8, 1, 8, 1, 8],
        #         [8, 8, 1, 8, 8, 1, 8, 8, 3, 1, 3, 1, 8],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        #         [8, 8, 1, 8, 8, 1, 8, 8, 3, 1, 3, 1, 8],
        #         [8, 8, 1, 8, 3, 1, 3, 8, 8, 1, 8, 1, 8],
        #         [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
        #         [8, 8, 1, 8, 3, 1, 3, 8, 8, 1, 8, 1, 8],
        #         [8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 8, 1, 8],
        #         [8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 8, 1, 8]
        #       ]
        #     },
        #     {
        #       "input": [
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 1, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 1, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9],
        #         [9, 9, 9, 9, 9, 9, 9, 9, 9]
        #       ],
        #       "output": [
        #         [9, 9, 9, 1, 9, 9, 1, 9, 9],
        #         [9, 9, 9, 1, 9, 9, 1, 9, 9],
        #         [9, 9, 3, 1, 3, 9, 1, 9, 9],
        #         [1, 1, 1, 2, 1, 1, 1, 1, 1],
        #         [9, 9, 3, 1, 3, 9, 1, 9, 9],
        #         [9, 9, 9, 1, 9, 3, 1, 3, 9],
        #         [1, 1, 1, 1, 1, 1, 2, 1, 1],
        #         [9, 9, 9, 1, 9, 3, 1, 3, 9],
        #         [9, 9, 9, 1, 9, 9, 1, 9, 9]
        #       ]
        #     },
        #     {
        #       "input": [
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7],
        #         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        #       ],
        #       "output": [
        #         [7, 7, 1, 7, 7, 7, 1, 3, 1, 3, 7],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        #         [7, 7, 1, 7, 7, 7, 1, 3, 1, 3, 7],
        #         [7, 3, 1, 3, 7, 7, 1, 7, 1, 7, 7],
        #         [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [7, 3, 1, 3, 7, 7, 1, 7, 1, 7, 7],
        #         [7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7],
        #         [7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7],
        #         [7, 7, 1, 7, 7, 3, 1, 3, 1, 7, 7],
        #         [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        #         [7, 7, 1, 7, 7, 3, 1, 3, 1, 7, 7]
        #       ]
        #     }
        #   ],
        #   "test": [
        #     {
        #       "input": [
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #         [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1]
        #       ],
        #       "output": [
        #         [8, 8, 1, 8, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [8, 8, 1, 8, 1, 8, 3, 1, 3, 8, 8, 8, 8, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
        #         [8, 8, 1, 8, 1, 8, 3, 1, 3, 8, 8, 8, 8, 1],
        #         [8, 3, 1, 3, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [8, 3, 1, 3, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [8, 8, 1, 8, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [8, 8, 1, 8, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [8, 8, 1, 8, 1, 8, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [8, 8, 1, 3, 1, 3, 8, 1, 8, 8, 8, 8, 8, 1],
        #         [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [8, 8, 1, 3, 1, 3, 8, 1, 8, 8, 8, 8, 3, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        #       ]
        #     }
        #   ]
        # }


        # Call the function
        description = generate_transformation_description(api_key, json_input)
        print(description)

        output_grid = apply_transformation(api_key, description, json_input["test"][0]['input'], json_input['train'])
        print("Generated Output Grid:")
        print(output_grid)
    except Exception as e:
        print(e)
        print("An error occurred. Please check the input data and try again.")
