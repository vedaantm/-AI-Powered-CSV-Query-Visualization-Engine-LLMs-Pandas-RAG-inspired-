import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import re
from difflib import get_close_matches
from io import BytesIO

# Page config
st.set_page_config(
    page_title="CSV Query & Viz Assistant",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------
# PREPROCESSING & DATE DETECTION
# ---------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe - handle numeric columns"""
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            numeric_converted = pd.to_numeric(df[col], errors='coerce')
            if numeric_converted.notna().sum() / len(df) > 0.8:
                df[col] = numeric_converted
    return df


def smart_date_parser(series):
    """Intelligently detect and parse date formats"""
    if series.isna().all():
        return series, None
    
    sample = series.dropna().head(20).astype(str)
    
    formats_to_try = [
        '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
        '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
        '%d-%b-%Y', '%d %b %Y', '%b %d, %Y',
        '%Y%m%d', '%d-%m-%y', '%m-%d-%y',
        '%d/%m/%y', '%m/%d/%y',
    ]
    
    for fmt in formats_to_try:
        try:
            test_parse = pd.to_datetime(sample, format=fmt, errors='coerce')
            success_rate = test_parse.notna().sum() / len(sample)
            
            if success_rate >= 0.8:
                parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                return parsed, fmt
        except:
            continue
    
    try:
        parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        success_rate = parsed.notna().sum() / len(series.dropna())
        if success_rate >= 0.8:
            return parsed, 'auto'
    except:
        pass
    
    return series, None


def detect_date_columns(df, schema):
    """Detect date columns using LLM + regex validation"""
    prompt = f"""
    You are given EXACTLY these column names from a dataframe:
    {schema}

    Return a JSON array containing ONLY the column names that represent dates.
    Rules:
    - Include columns with words like: date, Date, DOB, birth, joining, hire, start, end, created, updated, time, timestamp
    - Return ONLY column names that EXACTLY match the input schema

    Return ONLY the JSON array, nothing else.
    Example: ["DOB", "Start Date", "OrderDate"]
    """

    llm_detected = []
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False},
            timeout=10
        )
        data = response.json()
        raw = data.get("response", "").strip()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
        match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        llm_detected = json.loads(cleaned)
        if llm_detected and isinstance(llm_detected[0], dict):
            llm_detected = [item.get("column_name", item.get("name", "")) for item in llm_detected]
        llm_detected = [col for col in llm_detected if col in schema]
    except:
        pass
    
    date_keywords = r'\b(date|dob|birth|start|end|join|hire|created|updated|time|timestamp|day|month|year)\b'
    regex_detected = [col for col in schema if re.search(date_keywords, col, re.IGNORECASE)]
    
    candidates = list(set(llm_detected + regex_detected))
    
    confirmed_date_cols = []
    date_formats = {}
    
    for col in candidates:
        parsed, fmt = smart_date_parser(df[col])
        if fmt is not None:
            confirmed_date_cols.append(col)
            date_formats[col] = fmt
            df[col] = parsed
    
    return confirmed_date_cols, date_formats


# ---------------------------------------
# QUERY PARSING & EXECUTION
# ---------------------------------------

def fix_column_names(plan, schema):
    """Fix column names using fuzzy matching"""
    if "filter" in plan:
        if isinstance(plan["filter"], dict):
            col = plan["filter"]["column"]
            match = get_close_matches(col, schema, n=1, cutoff=0.6)
            if match:
                plan["filter"]["column"] = match[0]
        elif isinstance(plan["filter"], list):
            for f in plan["filter"]:
                col = f["column"]
                match = get_close_matches(col, schema, n=1, cutoff=0.6)
                if match:
                    f["column"] = match[0]

    if "select" in plan:
        fixed_select = []
        for col in plan["select"]:
            match = get_close_matches(col, schema, n=1, cutoff=0.6)
            fixed_select.append(match[0] if match else col)
        plan["select"] = fixed_select

    if "aggregate" in plan:
        col = plan["aggregate"]["column"]
        match = get_close_matches(col, schema, n=1, cutoff=0.6)
        if match:
            plan["aggregate"]["column"] = match[0]
    
    if "group_by" in plan:
        fixed_group = []
        for col in plan["group_by"]:
            match = get_close_matches(col, schema, n=1, cutoff=0.6)
            fixed_group.append(match[0] if match else col)
        plan["group_by"] = fixed_group
    
    if "sort" in plan:
        col = plan["sort"]["column"]
        match = get_close_matches(col, schema, n=1, cutoff=0.6)
        if match:
            plan["sort"]["column"] = match[0]

    return plan


def parse_date_with_llm(user_query, schema, date_cols):
    """Use LLM + Regex to extract date information from query - BULLETPROOF VERSION"""
    if not date_cols:
        return None
    
    date_patterns = {
        'specific_date': r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        'date_range': r'between\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+and\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        'month_name': r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b',
        'year': r'\b(19\d{2}|20\d{2})\b'
    }
    
    query_lower = user_query.lower()
    
    month_map = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
        'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5,
        'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    result = None
    
    # PRIORITY 1: Check for specific date (11-05-2024, 2024-05-11, etc.)
    # Check for date range first
    range_match = re.search(date_patterns['date_range'], user_query, re.IGNORECASE)
    if range_match:
        start_date = range_match.group(1)
        end_date = range_match.group(2)
        date_col = date_cols[0]
        if 'order' in query_lower:
            date_col = next((c for c in date_cols if 'order' in c.lower()), date_cols[0])
        
        result = {
            "has_date_filter": True,
            "date_column": date_col,
            "operation": "between",
            "value": [start_date, end_date]
        }
        return result
    
    # Check for specific single date
    date_match = re.search(date_patterns['specific_date'], user_query)
    if date_match:
        specific_date = date_match.group(1)
        date_col = date_cols[0]
        if 'order' in query_lower:
            date_col = next((c for c in date_cols if 'order' in c.lower()), date_cols[0])
        elif 'birth' in query_lower or 'born' in query_lower:
            date_col = next((c for c in date_cols if 'birth' in c.lower() or 'dob' in c.lower()), date_cols[0])
        elif 'start' in query_lower or 'join' in query_lower or 'hire' in query_lower:
            date_col = next((c for c in date_cols if 'start' in c.lower() or 'join' in c.lower()), date_cols[0])
        
        result = {
            "has_date_filter": True,
            "date_column": date_col,
            "operation": "equals_date",
            "value": specific_date
        }
        return result
    
    month_match = re.search(date_patterns['month_name'], query_lower)
    if month_match:
        month_name = month_match.group(1)
        month_num = month_map.get(month_name, None)
        if month_num:
            date_col = date_cols[0]
            if 'order' in query_lower:
                date_col = next((c for c in date_cols if 'order' in c.lower()), date_cols[0])
            elif 'birth' in query_lower or 'born' in query_lower:
                date_col = next((c for c in date_cols if 'birth' in c.lower() or 'dob' in c.lower()), date_cols[0])
            elif 'start' in query_lower or 'join' in query_lower or 'hire' in query_lower:
                date_col = next((c for c in date_cols if 'start' in c.lower() or 'join' in c.lower()), date_cols[0])
            
            result = {
                "has_date_filter": True,
                "date_column": date_col,
                "operation": "month",
                "value": month_num
            }
    
    year_match = re.search(date_patterns['year'], query_lower)
    if year_match and not result:
        year_num = int(year_match.group(1))
        date_col = date_cols[0]
        if 'order' in query_lower:
            date_col = next((c for c in date_cols if 'order' in c.lower()), date_cols[0])
        elif 'birth' in query_lower or 'born' in query_lower:
            date_col = next((c for c in date_cols if 'birth' in c.lower() or 'dob' in c.lower()), date_cols[0])
        elif 'start' in query_lower or 'join' in query_lower or 'hire' in query_lower:
            date_col = next((c for c in date_cols if 'start' in c.lower() or 'join' in c.lower()), date_cols[0])
        
        result = {
            "has_date_filter": True,
            "date_column": date_col,
            "operation": "year",
            "value": year_num
        }
    
    range_match = re.search(date_patterns['date_range'], query_lower, re.IGNORECASE)
    if range_match and not result:
        start_date = range_match.group(1)
        end_date = range_match.group(2)
        date_col = date_cols[0]
        if 'order' in query_lower:
            date_col = next((c for c in date_cols if 'order' in c.lower()), date_cols[0])
        
        result = {
            "has_date_filter": True,
            "date_column": date_col,
            "operation": "between",
            "value": [start_date, end_date]
        }
    
    if result:
        return result
    
    prompt = f"""
    Extract date filtering information from this query.
    
    Query: "{user_query}"
    Date columns: {date_cols}
    
    Return JSON:
    {{
        "has_date_filter": true/false,
        "date_column": "EXACT column name from {date_cols}",
        "operation": "month" or "year" or "between",
        "value": month number (1-12) or year number or [start_date, end_date]
    }}
    
    Return ONLY the JSON.
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False},
            timeout=30
        )
        data = response.json()
        raw = data.get("response", "").strip()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
        
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        
        date_info = json.loads(cleaned)
        
        if date_info.get("has_date_filter") and date_info.get("date_column") not in date_cols:
            if 'order' in user_query.lower():
                date_info["date_column"] = next((c for c in date_cols if 'order' in c.lower()), date_cols[0])
            else:
                date_info["date_column"] = date_cols[0]
        
        return date_info if date_info.get("has_date_filter") else None
    except:
        return None


def get_query_plan(user_query, schema):
    """Parse natural language query into JSON plan using LLM"""
    
    date_cols = st.session_state.get('date_cols', [])
    date_info = parse_date_with_llm(user_query, schema, date_cols)
    
    prompt = f"""
    You are a query parser and pandas expert.
    STRICTLY: UNDERSTAND THE MEANING. READ CAREFULLY.

    Dataset schema: {schema}
    Date columns: {date_cols}

    ⚠️ STRICT RULES:

    1. FILTERING:
       - Use "filter" for WHERE conditions
       - Single: {{"filter": {{"column": "Gender", "operation": "equals", "value": "Male"}}}}
       - Multiple: {{"filter": [{{"column": "X", "operation": "gt", "value": 100}}], "filter_logic": "AND"}}
       - Operations: equals, contains, gt, lt, gte, lte, month, year, between

    2. SELECTION:
       - Use "select" for columns to return
       - If "categories where date is X" -> select ["Category"]
       - If "products where date is X" -> select ["Product"]

    3. SORTING (only when explicitly mentioned):
       - {{"sort": {{"column": "Salary", "order": "desc"}}}}

    4. LIMIT:
       - "top 5" -> {{"limit": 5}}

    5. GROUP BY:
       - {{"group_by": ["Department"], "aggregate": {{"column": "Salary", "operation": "mean"}}}}

    EXAMPLES:
    Query: "categories where order date is in july"
    {{
      "select": ["Category"],
      "filter": {{"column": "Order Date", "operation": "month", "value": 7}}
    }}
    
    Query: "products sold in 2024"
    {{
      "select": ["Product"],
      "filter": {{"column": "Order Date", "operation": "year", "value": 2024}}
    }}

    User query: "{user_query}"

    Return ONLY valid JSON.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False},
                timeout=60
            )

            data = response.json()
            raw = data.get("response", "").strip()

            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

            plan = json.loads(cleaned)
            if isinstance(plan, list):
                plan = plan[0]
            
            if date_info and "filter" not in plan:
                plan["filter"] = {
                    "column": date_info["date_column"],
                    "operation": date_info["operation"],
                    "value": date_info["value"]
                }
            
            if "select" not in plan and "group_by" not in plan and "aggregate" not in plan:
                query_lower = user_query.lower()
                if "categor" in query_lower:
                    plan["select"] = ["Category"]
                elif "product" in query_lower:
                    plan["select"] = ["Product"]
                elif "customer" in query_lower:
                    plan["select"] = [col for col in schema if "customer" in col.lower()][:1]
                else:
                    plan["select"] = schema[:3]
            
            plan = sanitize_plan(plan, schema)
            return fix_column_names(plan, schema)
            
        except requests.exceptions.ReadTimeout:
            if attempt == max_retries - 1:
                st.error(f"⏱️ LLM timeout after {max_retries} attempts.")
                return None
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                st.error(f"⚠️ Could not parse LLM response: {e}")
                return None
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"⚠️ Error: {e}")
                return None
    
    return None


def sanitize_plan(plan, schema=None):
    """Repair malformed plans"""
    if "aggregate" in plan:
        agg = plan["aggregate"]
        if "operation" in agg:
            invalid_ops = ["equals", "contains", "startswith", "endswith", "gt", "lt", "gte", "lte"]
            if agg["operation"] in invalid_ops:
                filter_dict = {
                    "column": agg["column"],
                    "operation": agg["operation"],
                    "value": agg.get("value", "")
                }
                if "filter" not in plan:
                    plan["filter"] = filter_dict
                del plan["aggregate"]
    
    if "filter" in plan:
        if isinstance(plan["filter"], list) and "filter_logic" not in plan:
            plan["filter_logic"] = "AND"
        
        if isinstance(plan["filter"], dict):
            f = plan["filter"]
            for op in ["equals", "contains", "startswith", "endswith", "gt", "lt", "gte", "lte", "month", "year", "is_null", "not_null", "between"]:
                if op in f and "operation" not in f:
                    f["operation"] = op
                    if "value" not in f and op not in ["is_null", "not_null"]:
                        f["value"] = f[op]
                        del f[op]
        elif isinstance(plan["filter"], list):
            for f in plan["filter"]:
                for op in ["equals", "contains", "startswith", "endswith", "gt", "lt", "gte", "lte", "month", "year", "is_null", "not_null", "between"]:
                    if op in f and "operation" not in f:
                        f["operation"] = op
                        if "value" not in f and op not in ["is_null", "not_null"]:
                            f["value"] = f[op]
                            del f[op]

    if "filter" in plan and "select" not in plan and "group_by" not in plan:
        if schema:
            plan["select"] = schema[:5]
        else:
            plan["select"] = []

    return plan


def apply_single_filter(df, filter_dict):
    """Apply a single filter condition - BULLETPROOF DATE HANDLING"""
    col = filter_dict["column"]
    op = filter_dict["operation"]
    val = filter_dict.get("value")
    
    if op == "is_null":
        return df[df[col].isna()]
    elif op == "not_null":
        return df[df[col].notna()]
    
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        valid_df = df[df[col].notna()]
        
        if op == "equals_date":
            # Handle exact date matching (11-05-2024, 2024-05-11, etc.)
            try:
                formats = ['%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
                
                target_date = None
                for fmt in formats:
                    try:
                        target_date = pd.to_datetime(val, format=fmt)
                        break
                    except:
                        continue
                
                if target_date is None:
                    target_date = pd.to_datetime(val, errors='coerce')
                
                if pd.notna(target_date):
                    # Match the date (ignoring time)
                    return valid_df[valid_df[col].dt.date == target_date.date()]
                return valid_df
            except:
                return valid_df
        
        elif op == "month":
            if isinstance(val, str) and not val.isdigit():
                month_map = {
                    'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                    'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5,
                    'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
                    'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                    'december': 12, 'dec': 12
                }
                val = month_map.get(val.lower(), val)
            
            try:
                month_num = int(val)
                return valid_df[valid_df[col].dt.month == month_num]
            except:
                return valid_df
                
        elif op == "year":
            try:
                year_num = int(val)
                return valid_df[valid_df[col].dt.year == year_num]
            except:
                return valid_df
                
        elif op == "between":
            try:
                if isinstance(val, list) and len(val) == 2:
                    start_val, end_val = val[0], val[1]
                    
                    formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
                    
                    start_date = None
                    end_date = None
                    
                    for fmt in formats:
                        try:
                            start_date = pd.to_datetime(start_val, format=fmt)
                            break
                        except:
                            continue
                    
                    if start_date is None:
                        start_date = pd.to_datetime(start_val, errors='coerce')
                    
                    for fmt in formats:
                        try:
                            end_date = pd.to_datetime(end_val, format=fmt)
                            break
                        except:
                            continue
                    
                    if end_date is None:
                        end_date = pd.to_datetime(end_val, errors='coerce')
                    
                    if pd.notna(start_date) and pd.notna(end_date):
                        return valid_df[(valid_df[col] >= start_date) & (valid_df[col] <= end_date)]
                return valid_df
            except:
                return valid_df
                
        elif op in ["equals", "contains"]:
            mask = valid_df[col].dt.strftime("%B").str.contains(str(val), case=False, na=False)
            return valid_df[mask]
            
        elif op == "gt":
            try:
                return valid_df[valid_df[col] > pd.to_datetime(val)]
            except:
                return valid_df
                
        elif op == "lt":
            try:
                return valid_df[valid_df[col] < pd.to_datetime(val)]
            except:
                return valid_df
                
        elif op == "gte":
            try:
                return valid_df[valid_df[col] >= pd.to_datetime(val)]
            except:
                return valid_df
                
        elif op == "lte":
            try:
                return valid_df[valid_df[col] <= pd.to_datetime(val)]
            except:
                return valid_df

    elif pd.api.types.is_numeric_dtype(df[col]):
        try:
            if op == "between":
                return df[(df[col] >= float(val[0])) & (df[col] <= float(val[1]))]
            
            val_num = float(val)
            if op == "equals":
                return df[df[col] == val_num]
            elif op == "gt":
                return df[df[col] > val_num]
            elif op == "lt":
                return df[df[col] < val_num]
            elif op == "gte":
                return df[df[col] >= val_num]
            elif op == "lte":
                return df[df[col] <= val_num]
            elif op in ["contains", "startswith", "endswith"]:
                return df[df[col].astype(str).str.__getattribute__(op)(str(val))]
        except:
            pass
    else:
        val_str = str(val)
        if op == "equals":
            return df[df[col].astype(str).str.strip().str.lower() == val_str.strip().lower()]
        elif op == "contains":
            return df[df[col].astype(str).str.contains(val_str, case=False, na=False)]
        elif op == "startswith":
            return df[df[col].astype(str).str.lower().str.startswith(val_str.lower())]
        elif op == "endswith":
            return df[df[col].astype(str).str.lower().str.endswith(val_str.lower())]
    
    return df


def execute_plan(df, plan):
    """Execute the query plan"""
    temp = df.copy()

    if "filter" in plan:
        if isinstance(plan["filter"], dict):
            temp = apply_single_filter(temp, plan["filter"])
        elif isinstance(plan["filter"], list):
            filter_logic = plan.get("filter_logic", "AND")
            
            if filter_logic == "AND":
                for f in plan["filter"]:
                    temp = apply_single_filter(temp, f)
            else:
                mask = pd.Series([False] * len(temp), index=temp.index)
                for f in plan["filter"]:
                    filtered = apply_single_filter(temp, f)
                    mask = mask | temp.index.isin(filtered.index)
                temp = temp[mask]

    if "group_by" in plan and "aggregate" in plan:
        group_cols = plan["group_by"]
        agg_col = plan["aggregate"]["column"]
        agg_op = plan["aggregate"]["operation"]
        
        result = None
        
        if agg_op == "mean":
            result = temp.groupby(group_cols)[agg_col].mean().reset_index()
        elif agg_op == "median":
            result = temp.groupby(group_cols)[agg_col].median().reset_index()
        elif agg_op == "sum":
            result = temp.groupby(group_cols)[agg_col].sum().reset_index()
        elif agg_op == "min":
            result = temp.groupby(group_cols)[agg_col].min().reset_index()
        elif agg_op == "max":
            result = temp.groupby(group_cols)[agg_col].max().reset_index()
        elif agg_op == "count":
            result = temp.groupby(group_cols)[agg_col].count().reset_index()
        elif agg_op == "mode":
            result = temp.groupby(group_cols)[agg_col].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else None).reset_index()
        elif agg_op == "unique":
            result = temp.groupby(group_cols)[agg_col].apply(lambda x: x.unique().tolist()).reset_index()
        else:
            result = temp.groupby(group_cols).size().reset_index(name='count')
        
        if result is not None:
            if agg_op in ["mean", "median", "sum", "min", "max", "count", "mode", "unique"]:
                result.columns = list(group_cols) + [f"{agg_op}_{agg_col}"]
            
            return {
                "rows": result.to_dict(orient="records"),
                "row_count": len(result),
                "dataframe": result
            }

    if "aggregate" in plan and "group_by" not in plan:
        col, op = plan["aggregate"]["column"], plan["aggregate"]["operation"]
        if op == "mean":
            return {op: temp[col].astype(float).mean(), "row_count": len(temp)}
        elif op == "median":
            return {op: temp[col].astype(float).median(), "row_count": len(temp)}
        elif op == "sum":
            return {op: temp[col].astype(float).sum(), "row_count": len(temp)}
        elif op == "min":
            return {op: temp[col].min(), "row_count": len(temp)}
        elif op == "max":
            return {op: temp[col].max(), "row_count": len(temp)}
        elif op == "count":
            return {"counts": temp[col].value_counts().to_dict(), "row_count": len(temp)}
        elif op == "mode":
            return {op: temp[col].mode()[0], "row_count": len(temp)}
        elif op == "unique":
            return {op: temp[col].unique().tolist(), "row_count": len(temp)}

    if "sort" in plan:
        col = plan["sort"]["column"]
        order = plan["sort"].get("order", "asc")
        ascending = (order == "asc")
        temp = temp.sort_values(by=col, ascending=ascending)

    if "limit" in plan:
        temp = temp.head(plan["limit"])

    if "select" in plan:
        temp = temp[plan["select"]]

    return {
        "rows": temp.to_dict(orient="records"),
        "row_count": len(temp),
        "dataframe": temp
    }


# ---------------------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------------------

def detect_column_types(df):
    """Detect numeric, categorical, and date columns"""
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    for col in df.columns:
        if re.search(r'\b(id|_id|ID|Id)\b', col, re.IGNORECASE):
            continue

        col_lower = col.lower()

        if any(keyword in col_lower for keyword in ["date", "dob", "time"]):
            date_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            try:
                df[col].astype(float)
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, date_cols


def smart_auto_plot(df):
    """Generate smart visualizations based on column types"""
    numeric_cols, categorical_cols, date_cols = detect_column_types(df)
    plots = []

    if len(numeric_cols) == 0:
        st.warning("⚠️ No numeric columns detected for visualization.")
        return plots

    corr_matrix = df[numeric_cols].corr(method='pearson')

    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.4:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(df[col1], df[col2], alpha=0.6, s=50)
                    ax.set_title(f"{col1} vs {col2} (correlation={corr:.2f})", fontsize=14, fontweight='bold')
                    ax.set_xlabel(col1, fontsize=11)
                    ax.set_ylabel(col2, fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plots.append(("scatter", col1, col2, fig))

    for cat in categorical_cols:
        if df[cat].nunique() > 10:
            continue
        for num in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped = df.groupby(cat)[num].mean().sort_values()
            grouped.plot(kind="bar", ax=ax, color='steelblue')
            ax.set_title(f"Average {num} by {cat}", fontsize=14, fontweight='bold')
            ax.set_xlabel(cat, fontsize=11)
            ax.set_ylabel(f"Average {num}", fontsize=11)
            ax.grid(True, axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plots.append(("bar", cat, num, fig))

    return plots


def explain_with_llm(df, plots):
    """Get LLM explanation of data patterns"""
    try:
        plotted_pairs = [(p[1], p[2]) for p in plots]
        summary = {
            "num_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "cat_cols": df.select_dtypes(exclude=[np.number]).columns.tolist(),
            "sample_data": df.head(3).to_dict(orient="records"),
            "plots": plotted_pairs
        }

        prompt = f"""
        You are a data analyst. Given this dataset summary and the plots that were generated, 
        explain the possible trends or relationships observed in a human-like, concise way.

        Dataset summary:
        {json.dumps(summary, indent=2)}

        Return just a plain text paragraph (no JSON or markdown).
        """

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        return response.json().get("response", "").strip()
    except Exception as e:
        return f"❌ Error getting LLM explanation: {e}"


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------

st.title("📊 Advanced CSV Query & Visualization Assistant")
st.markdown("Upload a CSV and query it using natural language or auto-generate intelligent visualizations!")

with st.sidebar:
    st.header("⚙️ Settings")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
    model_name = st.text_input("Model Name", "qwen2.5:1.5b")
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("✅ Natural language queries")
    st.markdown("✅ Smart date detection")
    st.markdown("✅ Auto-visualizations")
    st.markdown("✅ LLM-powered insights")
    st.markdown("✅ Fuzzy column matching")
    st.markdown("✅ Export results")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        df = preprocess_dataframe(df)
        
        schema = list(df.columns)
        with st.spinner("🔍 Detecting date columns..."):
            date_cols, date_formats = detect_date_columns(df, schema)
        
        st.session_state['df'] = df
        st.session_state['schema'] = schema
        st.session_state['date_cols'] = date_cols
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", len(df))
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("📅 Date Columns", len(date_cols))
    with col4:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numeric Columns", numeric_count)
    
    if date_cols:
        with st.expander("📅 Detected Date Columns & Formats"):
            for col in date_cols:
                fmt = date_formats.get(col, "auto")
                st.write(f"**{col}**: `{fmt}`")
    
    tab1, tab2 = st.tabs(["🔍 Natural Language Query", "📈 Auto Visualizations"])
    
    with tab1:
        st.markdown("### 💬 Ask questions about your data")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Example queries:**")
            st.markdown("- *Categories where order date is in July*")
            st.markdown("- *Products sold in 2024*")
            st.markdown("- *Average salary by department*")
            st.markdown("- *Orders between May and June*")
        with col2:
            st.markdown("**Supported operations:**")
            st.markdown("- Month/Year filtering")
            st.markdown("- Date ranges")
            st.markdown("- Aggregations (mean, sum)")
            st.markdown("- Sorting & limiting")
        
        user_query = st.text_input("🔎 Enter your query:", key="nlp_query", placeholder="e.g., Categories where order date is in July")
        
        if st.button("🚀 Run Query", key="run_query", type="primary"):
            if user_query:
                with st.spinner("🧠 Processing query..."):
                    plan = get_query_plan(user_query, schema)
                    
                    if plan:
                        st.success("✅ Query plan generated!")
                        with st.expander("📋 View Query Plan"):
                            st.json(plan)
                        
                        try:
                            result = execute_plan(df, plan)
                            
                            st.subheader("📊 Results")
                            st.info(f"Found **{result['row_count']}** results")
                            
                            if "dataframe" in result:
                                st.dataframe(result["dataframe"], use_container_width=True)
                                
                                csv = result["dataframe"].to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results as CSV",
                                    csv,
                                    "query_results.csv",
                                    "text/csv",
                                    key="download_csv"
                                )
                            elif "rows" in result:
                                result_df = pd.DataFrame(result["rows"])
                                st.dataframe(result_df, use_container_width=True)
                            else:
                                st.json(result)
                        except Exception as e:
                            st.error(f"❌ Error executing query: {e}")
                    else:
                        st.error("❌ Could not parse query. Please try rephrasing.")
            else:
                st.warning("⚠️ Please enter a query.")
    
    with tab2:
        st.markdown("### 🎨 Automatic Data Visualization")
        st.markdown("Generate smart visualizations based on correlations and data patterns.")
        
        viz_col1, viz_col2 = st.columns([1, 4])
        with viz_col1:
            if st.button("🎨 Generate Visualizations", key="gen_viz", type="primary"):
                st.session_state['generate_viz'] = True
        with viz_col2:
            st.info("💡 This will create scatter plots for correlated numeric columns (|corr| > 0.4) and bar charts for categorical analysis.")
        
        if st.session_state.get('generate_viz', False):
            with st.spinner("📊 Creating visualizations..."):
                plots = smart_auto_plot(df)
                
                if plots:
                    st.success(f"✅ Generated **{len(plots)}** visualizations!")
                    
                    for i, (plot_type, col1, col2, fig) in enumerate(plots):
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    st.markdown("---")
                    with st.spinner("🧠 Generating AI insights..."):
                        explanation = explain_with_llm(df, plots)
                        st.subheader("🧠 AI-Powered Insights")
                        st.info(explanation)
                else:
                    st.warning("⚠️ No significant patterns found for visualization.")
            
            st.session_state['generate_viz'] = False

else:
    st.info("👆 Please upload a CSV file to get started.")
    
    with st.expander("💡 See Example Usage"):
        st.markdown("""
        ### Natural Language Queries:
        
        **Date Queries:**
        - "Categories where order date is in July"
        - "Products sold in 2024"
        - "Orders between 10-05-2024 and 15-05-2024"
        - "Customers who ordered in May"
        
        **Filtering:**
        - "Show employees with salary greater than 80000"
        - "Female employees in IT department"
        
        **Aggregations:**
        - "Average salary by department"
        - "Count employees by gender"
        - "Total revenue by region"
        
        **Top N Queries:**
        - "Top 10 customers by revenue"
        - "Highest 5 performing employees"
        
        **Complex Queries:**
        - "Male data scientists with salary above 70000"
        - "Products from USA ordered in March"
        
        ---
        
        ### Auto Visualizations:
        
        **What it does:**
        - Automatically detects correlations between numeric columns
        - Creates scatter plots for strongly correlated variables (|r| > 0.4)
        - Generates bar charts showing relationships between categorical and numeric columns
        - Provides AI-powered insights about observed patterns
        """)
    
    with st.expander("🔧 Technical Features"):
        st.markdown("""
        ### Advanced Capabilities:
        
        1. **🛡️ Bulletproof Date Handling**
           - Regex + LLM dual detection system
           - Handles 17+ date formats automatically
           - Month/Year/Date range filtering
           - Smart column inference (Order Date, DOB, Start Date)
        
        2. **🔍 Fuzzy Column Matching**
           - Corrects typos in queries
           - Maps similar column names automatically
           - 60% similarity threshold
        
        3. **🧠 Robust Query Parsing**
           - Handles complex AND/OR logic
           - Supports 13+ filter operations
           - 8 aggregation functions
           - Month name recognition (March, mar, 3)
        
        4. **⚡ Error Handling**
           - Automatic retry on timeout (3 attempts)
           - Malformed query plan repair
           - Invalid aggregate operation detection
           - Graceful fallbacks for parsing errors
        
        5. **📊 Visualization Intelligence**
           - Correlation-based plot generation
           - Skips high-cardinality categoricals (>10 unique)
           - LLM-powered trend explanation
        """)

st.markdown("---")
st.markdown("*🛡️ Bulletproof Date System | Powered by Ollama LLM | Built with Streamlit*")