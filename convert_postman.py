import json
import os

json_path = r"c:\Users\S.JUNAID\Downloads\4_5767339968343056061.json"

def parse_postman_items(items):
    requests = []
    for item in items:
        if "item" in item:
            requests.extend(parse_postman_items(item["item"]))
        elif "request" in item:
            req = item["request"]
            
            # Extract URL
            url_obj = req.get("url")
            url = ""
            if isinstance(url_obj, str):
                url = url_obj
            elif isinstance(url_obj, dict):
                url = url_obj.get("raw", "")
            
            # Extract Headers
            headers = {}
            if isinstance(req.get("header"), list):
                for h in req.get("header"):
                    headers[h["key"]] = h["value"]
            
            # Extract Body
            body = ""
            body_obj = req.get("body")
            if body_obj and body_obj.get("mode") == "raw":
                body = body_obj.get("raw", "")
            
            requests.append({
                "name": item["name"],
                "method": req.get("method", "GET"),
                "url": url,
                "headers": headers,
                "body": body
            })
    return requests

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = data.get("item", [])
    requests = parse_postman_items(items)
    print("POSTMAN_REQUESTS_OVERRIDE = " + json.dumps(requests, indent=4))

except Exception as e:
    print(f"Error: {e}")
