import os
import re
import pdb
def get_top_5_max_accuracy_from_txt_files(folder_path):
    """
    从最近更新的20个 .txt 文件中提取 INFO Max accuracy 的值，返回前5个最大值及对应的文件名。

    参数:
    - folder_path (str): 文件夹路径

    返回:
    - None
    """
    if not os.path.isdir(folder_path):
        print(f"错误: {folder_path} 不是一个有效的文件夹路径。")
        return

    # 获取最近更新的20个 .txt 文件
    txt_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.out')],
        key=lambda x: os.path.getmtime(os.path.join(folder_path, x)),
        reverse=True
    )

    accuracy_results = []

    # 正则表达式匹配 INFO Max accuracy
    pattern = re.compile(r'INFO Max accuracy: (\d+\.\d+)')
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    match = pattern.search(line)
                    if match:
                        accuracy = float(match.group(1))
                        accuracy_results.append((txt_file, accuracy))
                        break  # 只取最后一个匹配的 INFO Max accuracy
        except Exception as e:
            print(f"读取文件 {txt_file} 出错: {e}")

    # 按 accuracy 排序，取前5个
    top_5 = sorted(accuracy_results, key=lambda x: x[1], reverse=True)[:10]

    # 打印结果
    print("\n🔝 **前5个最大 INFO Max accuracy 的文件:**")
    for i, (file, acc) in enumerate(top_5, 1):
        print(f"{i}. 文件名: {file}, Max accuracy: {acc}")


# 示例用法
folder_path = os.getcwd()  # 当前文件夹路径
get_top_5_max_accuracy_from_txt_files(folder_path)