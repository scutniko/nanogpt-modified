"""
批量重命名 .npy 文件为 .txt 后缀
"""
import os
from pathlib import Path

def rename_npy_to_txt(directory="./edu_fineweb10B"):
    """
    将指定目录下的所有 .npy 文件重命名为 .txt 后缀
    
    参数:
        directory: 目标目录路径（默认 ./edu_fineweb10B）
    """
    target_dir = Path(directory)
    
    if not target_dir.exists():
        print(f"❌ 错误：目录 {directory} 不存在")
        return
    
    # 获取所有 .npy 文件
    npy_files = list(target_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"在 {directory} 目录下没有找到 .npy 文件")
        return
    
    print(f"找到 {len(npy_files)} 个 .npy 文件")
    print("=" * 60)
    
    # 询问用户确认
    print("\n即将重命名以下文件：")
    for i, file in enumerate(sorted(npy_files)[:5], 1):
        print(f"  {i}. {file.name} -> {file.stem}.txt")
    
    if len(npy_files) > 5:
        print(f"  ... 以及其他 {len(npy_files) - 5} 个文件")
    
    print("\n" + "=" * 60)
    response = input("确认重命名？(y/n): ").strip().lower()
    
    if response != 'y':
        print("操作已取消")
        return
    
    # 执行重命名
    print("\n开始重命名...")
    success_count = 0
    
    for npy_file in sorted(npy_files):
        try:
            # 构建新文件名
            new_name = npy_file.with_suffix('.txt')
            
            # 检查目标文件是否已存在
            if new_name.exists():
                print(f"⚠️  跳过 {npy_file.name}（目标文件已存在）")
                continue
            
            # 重命名
            npy_file.rename(new_name)
            success_count += 1
            print(f"✓ {npy_file.name} -> {new_name.name}")
            
        except Exception as e:
            print(f"❌ 重命名 {npy_file.name} 时出错: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ 完成！成功重命名 {success_count}/{len(npy_files)} 个文件")
    print("=" * 60)

if __name__ == "__main__":
    rename_npy_to_txt("./edu_fineweb10B")

