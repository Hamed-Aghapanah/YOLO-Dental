import pandas as pd
import openpyxl

def process_excel(file_path):
    # خواندن فایل اکسل
    df = pd.read_excel(file_path)
    
    # تبدیل حروف به اعداد و محاسبه مجموع
    letter_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    
    for index, row in df.iterrows():
        total = 0
        for col in ['impaction', 'ramus relation', 'angulation', 'root morphology', 'root curvture']:
            letter = str(row[col]).strip().upper()
            total += letter_to_num.get(letter, 0)
        
        # ذخیره مجموع در ستون degree of hardnes
        df.at[index, 'degree of hardnes'] = total
        
        # تعیین class_label بر اساس مجموع
        if 0 <= total <= 8:
            class_label = 'A'
        elif 8 < total <= 11:
            class_label = 'B'
        else:
            class_label = 'C'
        
        # اضافه کردن ستون class_label اگر وجود ندارد
        if 'class_label' not in df.columns:
            df['class_label'] = ''
        
        df.at[index, 'class_label'] = class_label
    
    # ذخیره فایل پردازش شده
    output_path = file_path.replace('.xlsx', '_processed.xlsx')
    df.to_excel(output_path, index=False)
    print(f"فایل پردازش شده با موفقیت در {output_path} ذخیره شد.")

# اجرای تابع با مسیر فایل
process_excel('class.xlsx')