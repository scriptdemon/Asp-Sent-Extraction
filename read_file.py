import json
import os

file = open('mapped_final_data1.txt','r')
for line in file:
    json_obj = json.loads(line)
    product_id = str(json_obj['asin'])
    if not os.path.exists("aspect_sentiment_pairs/"+product_id+".txt"):
        open("aspect_sentiment_pairs/"+product_id+".txt","w").close()

