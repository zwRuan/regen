import sys
import os
import argparse

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 打开输出文件用于写入合并后的内容
    with open(args.output_file, 'w') as outfile:
        # 遍历输入目录中的所有文件
        for i in range(5):
            filename = os.path.join(args.input_dir, f'predictions_{i}.jsonl')
            if os.path.exists(filename):
                with open(filename, 'r') as infile:
                    outfile.write(infile.read())

if __name__ == '__main__':
    main()
