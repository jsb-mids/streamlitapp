import pandas as pd

furniture_df = pd.read_csv('./data/cleaned_target_furniture_dataset.csv')

def get_product_details(uniq_ids):
    product_details_list = []
    for uniq_id in uniq_ids:
        product_details = furniture_df[furniture_df['uniq_id'] == uniq_id].iloc[0]
        product_dict = {
            'uniq_id': uniq_id,
            'title': product_details['title'],
            'url': product_details['main_image'],
            'price': product_details['price']
        }
        product_details_list.append(product_dict)
    return product_details_list