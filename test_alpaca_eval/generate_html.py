import json
def read_jsons(file_name: str):
    dict_objs = []
    with open(file_name, "r") as f:
        for line in f:
            dict_objs.append(json.loads(line))
    return dict_objs

def generate_html(data, output_file="debug/html/model_comparison.html", max_num=200):

    data = data[:max_num]

    # Function to escape HTML special characters
    def escape_html(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Function to wrap <THOUGHT> tags with gray styling while keeping the tokens
    def style_thoughts(text):
        return text.replace(
            "<THOUGHT>", "<span style='color:gray'>&lt;THOUGHT&gt;"
        ).replace(
            "</THOUGHT>", "&lt;/THOUGHT&gt;</span>"
        ).replace(
            "&lt;THOUGHT&gt;", "<span style='color:gray'>&ltTHOUGHT&gt;"
        ).replace(
            "&lt;/THOUGHT&gt;", "&lt;/THOUGHT&gt;</span>"
        )

    for item in data:
        item["instruction"] = item["prompt"]
        item['reference'] = style_thoughts(escape_html(item.get('label', '')))
        item['generation'] = style_thoughts(escape_html(item.get('predict', '')))

    # Generate HTML content
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Comparison</title>
        <style>
            .container {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
            }}
            .box {{
                width: 45%;
                border: 1px solid black;
                padding: 10px;
                overflow: auto;
                max-height: 500px;
                white-space: pre-wrap;
                font-family: monospace;
            }}
            .prompt {{
                text-align: center;
                font-weight: bold;
                margin-bottom: 20px;
                white-space: pre-wrap;
                font-family: monospace;
            }}
            .file-path {{
                text-align: center;
                font-size: 12px;
                color: gray;
                margin-top: -10px;  /* adjust as needed to bring closer to the boxes */
            }}
            .navigation {{
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                padding: 10px;
                background-color: #f9f9f9;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div id="app">
            <div class="prompt" id="prompt">Prompt: </div>
            <div class="container">
                <div class="box" id="boxA">Model A Response</div>
                <div class="box" id="boxB">Model B Response</div>
            </div>
            <div class="container">
                <div class="file-path">Reference</div>
                <div class="file-path">Generation</div>
            </div>
            <div class="navigation">
                <button onclick="previousSample()">Previous</button>
                <span id="sampleCounter">1 / {total_samples}</span>
                <button onclick="nextSample()">Next</button>
            </div>
        </div>

        <script>
            const data = {data};
            let currentIndex = 0;

            function updateContent(index) {{
                document.getElementById('prompt').innerText = "Prompt: " + data[index].instruction;
                document.getElementById('boxA').innerHTML = data[index].reference;
                document.getElementById('boxB').innerHTML = data[index].generation;
                document.getElementById('sampleCounter').innerText = (index + 1) + " / " + data.length;
            }}

            function previousSample() {{
                if (currentIndex > 0) {{
                    currentIndex--;
                    updateContent(currentIndex);
                }}
            }}

            function nextSample() {{
                if (currentIndex < data.length - 1) {{
                    currentIndex++;
                    updateContent(currentIndex);
                }}
            }}

            // Add event listener for left and right arrow keys
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'ArrowLeft') {{
                    previousSample();
                }} else if (event.key === 'ArrowRight') {{
                    nextSample();
                }}
            }});

            // Initialize content
            updateContent(currentIndex);
        </script>
    </body>
    </html>
    '''.format(
        data=json.dumps(data),
        total_samples=len(data)
    )

    # Write the generated HTML content to the specified file
    with open(output_file, 'w', encoding='utf-8') as output_file_handle:
        output_file_handle.write(html_content)

    print("HTML file has been generated:", output_file)


import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = os.path.dirname(args.in_file)
    else:
        output_dir = args.output_dir

    data = read_jsons(args.in_file)
    os.makedirs(output_dir, exist_ok=True)
    generate_html(data, os.path.join(output_dir, f"case_study.html"))