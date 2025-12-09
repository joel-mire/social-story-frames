from ssf.Constants import *
import html

def fmt_text_for_potato(row, plausibility_rating_task=True):
    context_dict = _fmt_conversational_context(row)
    
    context_header_color = "#2e7d32"  # Green for context header
    context_bg_color = "#e0f2e0"      # Lighter green for context background
    storytelling_header_color = "#1e3a8a"  # Dark blue for storytelling
    storytelling_bg_color = "#e6eeff"  # Light blue for storytelling background
    
    # Start with a container div for the two-column layout
    html_output = '<div style="display: flex; flex-direction: column; gap: 15px; font-family: Arial, sans-serif;">'

    # ----- ROW: Content Columns -----
    html_output += '<div style="display: flex; gap: 15px;">'

    # ----- LEFT COLUMN (Context) -----
    html_output += '<div style="flex: 1;">'
    html_output += f'<div style="background-color: {context_header_color}; color: white; padding: 8px 12px; font-weight: bold; border-radius: 4px 4px 0 0; font-size: 16px;">Conversational Context</div>'
    html_output += f'<div style="background-color: {context_bg_color}; padding: 10px; border-radius: 0 0 4px 4px;">'

    key_colors = {
        "subreddit": "#1A8FE3",
        "subreddit description": "#8A4FD0",
        "subreddit norms": "#00B074",
        "initial post summary": "#FF8C29",
        "conversation history": "#E74C3C"
    }

    for key, value in context_dict.items():
        color = key_colors.get(key, "#1A8FE3")
        html_output += f"""
        <div style="margin-bottom: 10px;">
          <div style="display: inline-block; padding: 4px 10px; background-color: {color}; color: white; font-weight: 500; border-radius: 4px; margin-bottom: 4px; font-size: 14px;">
            {key}
          </div>
          <div style="margin-top: 3px; color: #374151; padding-left: 4px; font-size: 14px;">
            {value}
          </div>
        </div>
        """

    html_output += '</div></div>'

    # ----- RIGHT COLUMN (Storytelling) -----
    html_output += '<div style="flex: 1;">'
    html_output += f'<div style="background-color: {storytelling_header_color}; color: white; padding: 8px 12px; font-weight: bold; border-radius: 4px 4px 0 0; font-size: 16px;">Storytelling Text</div>'
    html_output += f'<div style="background-color: {storytelling_bg_color}; padding: 10px; border-radius: 0 0 4px 4px; font-size: 18px;">{html.escape(row.get("_text", ""))}</div>'
    html_output += '</div>'

    # Close the row of columns
    html_output += '</div>'

    # ----- Full-Width Instructions -----
    html_output += '<hr style="width: 100%; margin: 15px 0;">'
    
    if plausibility_rating_task:
        html_output += '<div style="font-size: 16px; color: #374151; width: 100%;"><em><strong>Task</strong>: Based on the conversational context and the storytelling text, rate the plausibility of the questions below. A plausible statement seems reasonable, believable, or probable based on the available evidence, even if it is not certain.</em></div>'
    else: # human-written task
        html_output += '<div style="font-size: 16px; color: #374151; width: 100%;"><em><strong>Task</strong>: Based on the conversational context and the storytelling text, answer the questions below. Each answer below should focus on one main idea. If you have multiple ideas, choose the one your most confident in and answer based on that, rather than listing multiple ideas in a single answer. Complete all templates by replacing the text between {{ and }} with short answers. <strong>***NEVER EDIT TEXT OUTSIDE THE DOUBLE BRACES.***</strong> Ensure every template is fully answered. Hover your mouse over the questions to see brief general background information about the question, which may help you quickly brainstorm if you are stuck. </em></div>'

    # Close the main container
    html_output += '</div>'

    return html_output

def _fmt_conversational_context(row):
  subreddit_name = row[COMMUNITY_META_COL]
  subreddit_description = row[COMMUNITY_DESCRIPTION_META_COL]
  subreddit_values = row[COMMUNITY_VALUES_META_COL]
  progenitor_summary = row[PROGENITOR_SUMMARY_META_COL]
  conversation_summary = row[CONVERSATION_SUMMARY_META_COL]
  out = {}
  if subreddit_name:
      out['subreddit'] = subreddit_name
  if subreddit_description:
      out['subreddit description'] = subreddit_description
  if subreddit_description:
      out['subreddit norms'] = subreddit_values
  if progenitor_summary:
      out['initial post summary'] = progenitor_summary
  if progenitor_summary:
      out['conversation history'] = conversation_summary
  return out

def _split_keep_delimiter(text, delimiter, maxsplit=1):
    parts = text.split(delimiter, maxsplit)
    return delimiter + parts[1]

def get_label_suggestions(taxonomy):
    return {f"{dim}":_split_keep_delimiter(template, "{", maxsplit=1) for dim, template in taxonomy.dim_templates_dict.items()}