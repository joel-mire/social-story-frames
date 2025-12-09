import pandas as pd
from collections import defaultdict
import re
import os
from textwrap import dedent

LATEX_ESCAPES = {
  '&':  r'\&',
  '%':  r'\%',
  '$':  r'\$',
  '#':  r'\#',
  '_':  r'\_',
  '{':  r'\{',
  '}':  r'\}',
  '~':  r'\textasciitilde{}',
  '^':  r'\textasciicircum{}',
  '\\': r'\textbackslash{}',
}

def _split_keep_delimiter(text, delimiter, maxsplit=1):
  parts = text.split(delimiter, maxsplit)
  return delimiter + parts[1]

class Taxonomy:
  # File name constants
  TAXONOMY_CSV = "taxonomy.csv"
  SUMMARIES_CSV = "taxonomy_summaries.csv"
  TEMPLATES_CSV = "taxonomy_templates.csv"

  # Regex pattern for template variables
  TEMPLATE_VAR_PATTERN = re.compile(r"\{\{.*?\}\}")

  def __init__(self, taxonomy_dir):
    self.taxonomy_dir = taxonomy_dir
    tax_path = f"{taxonomy_dir}/{self.TAXONOMY_CSV}"
    self.tax_df = pd.read_csv(tax_path)
    self.dim_data_dict = self._build_dim_data_dict(self.tax_df)

    tax_summaries_path = f"{taxonomy_dir}/{self.SUMMARIES_CSV}"
    tax_summaries_df = pd.read_csv(tax_summaries_path)
    self.dim_summaries_dict = self._build_dim_summaries_dict(tax_summaries_df)

    tax_templates_path = f"{taxonomy_dir}/{self.TEMPLATES_CSV}"
    tax_templates_df = pd.read_csv(tax_templates_path)
    self.dim_templates_dict = self._build_dim_templates_dict(tax_templates_df)

    self.dim_vars_dict = self.get_template_var_dict()

  def get_dim_vars_dict(self):
    return self.dim_vars_dict
  
  def _build_dim_data_dict(self, tax_df):
    dim_data_dict = defaultdict(lambda: defaultdict(dict))
    for _, row in tax_df.iterrows():
      dim_data_dict[row['dimension']][row['category']] = {
          "definition": row['definition'],
          "example": row['example']
      }
    return dim_data_dict

  def get_excluded_categories(self, dim):
    """Get categories that should be excluded from taxonomy classification for a dimension."""
    excluded = self.tax_df[
      (self.tax_df['dimension'] == dim) & 
      (self.tax_df['include_in_tax_class'] == False)
    ]['category'].tolist()
    return excluded

  def _build_dim_summaries_dict(self, tax_summaries_df):
    return dict(zip(tax_summaries_df['dimension'], tax_summaries_df['summary']))

  def _build_dim_templates_dict(self, tax_templates_df):
    return dict(zip(tax_templates_df['dimension'], tax_templates_df['template']))

  def count_dim_vars(self, dim):
    dim_template = self.dim_templates_dict[dim]
    matches = self.TEMPLATE_VAR_PATTERN.findall(dim_template)
    return len(matches)

  def get_static_template_segments(self, dim):
    dim_template = self.dim_templates_dict[dim]
    return self.TEMPLATE_VAR_PATTERN.split(dim_template)

  def get_template_prefix(self, dim):
    return self.get_static_template_segments(dim)[0]
  
  def get_var_vals(self, dim, response):
    static_template_segments = self.get_static_template_segments(dim)
    var_vals = []
    try:
      for i in range(len(static_template_segments) - 1):
        antecedent_static_segment = static_template_segments[i]
        subsequent_static_segment = static_template_segments[i + 1]
        var = self.get_substring_between(response, antecedent_static_segment, subsequent_static_segment)
        var_vals.append(var)
    except Exception as e:
      # print(f"WARNING: unable to extract dim={dim} var vals. Response: {response}. Exception: {e}")
      var_vals = ["ERROR"]
    return var_vals
      
  def get_substring_between(self, s, antecedent_substring, subsequent_substring):
    pattern = re.escape(antecedent_substring) + r'(.*)' + re.escape(subsequent_substring)
    match = re.search(pattern, s)
    if match:
      return match.group(1)
    else:
      raise ValueError(f"No match found between '{antecedent_substring}' and '{subsequent_substring}' in the string: {s}")

  def write_tooltips(self, out_dir):
    for dim, summary in self.dim_summaries_dict.items():
      clipped_summary = summary.rsplit("\n", 1)[0]
      out_path = os.path.join(out_dir, f"{dim}.html")
      html_content = f"<p>{clipped_summary}</p>"
      with open(out_path, "w") as f:
        f.write(html_content)

  def get_dims(self):
    dims = list(self.dim_data_dict.keys())
    return dims

  def extract_text_within_braces(self, text):
    # Note: Using a different pattern that captures content inside braces
    pattern = re.compile(r"{{(.*?)}}")
    extracted_text = pattern.findall(text)
    return extracted_text

  def get_template_var_dict(self):
    dim_vars = {}
    for dim, template in self.dim_templates_dict.items():
      dim_vars[dim] = self.extract_text_within_braces(template)
    return dim_vars

  def get_label_suggestions(self):
    return {f"{dim}":_split_keep_delimiter(template, "{", maxsplit=1) for dim, template in self.dim_templates_dict.items()}

  @classmethod
  def _escape_latex(cls, text):
    """
    Replace characters that would otherwise break LaTeX compilation.
    """
    if not isinstance(text, str):
      text = str(text)
    return re.sub(
      '|'.join(re.escape(k) for k in LATEX_ESCAPES),
      lambda m: LATEX_ESCAPES[m.group()],
      text
    )

  def to_latex_table(self,
                      path,
                      caption="Taxonomy of pragmatic aspects of narrative intent and reception of storytelling on social media",
                      label="tab:taxonomy"):
    """
    Return a self-contained LaTeX longtable containing every dimension,
    sub-dimension, definition, and example stored in this Taxonomy
    instance.
    """
    # Define column widths that add up to 0.9\textwidth total
    col_def = dedent(r"""
      p{0.2\textwidth}
      p{0.3\textwidth}
      p{0.4\textwidth}
    """).strip().replace("\n", "")

    lines = [
      r"\begingroup",
      r"\scriptsize",
      r"\setlength{\LTcapwidth}{0.9\textwidth}",
      rf"\begin{{longtable}}[width=0.9\textwidth]{{{col_def}}}",
      rf"\caption{{{self._escape_latex(caption)}}}",
      rf"\label{{{label}}} \\",
      r"\toprule",
      r"\rowcolor[gray]{0.75}",
      r"\multicolumn{1}{p{0.2\textwidth}}{\textbf{Sub-dimension}} & \multicolumn{1}{p{0.3\textwidth}}{\textbf{Definition}} & \multicolumn{1}{p{0.4\textwidth}}{\textbf{Example}} \\",
      r"\midrule",
      r"\endfirsthead",
      r"\multicolumn{3}{p{0.9\textwidth}}{\tablename\ \thetable{} -- continued from previous page} \\",
      r"\toprule",
      r"\rowcolor[gray]{0.75}",
      r"\multicolumn{1}{p{0.2\textwidth}}{\textbf{Sub-dimension}} & \multicolumn{1}{p{0.3\textwidth}}{\textbf{Definition}} & \multicolumn{1}{p{0.4\textwidth}}{\textbf{Example}} \\",
      r"\midrule",
      r"\endhead",
      r"\midrule",
      r"\multicolumn{3}{r}{Continued on next page} \\",
      r"\endfoot",
      r"\bottomrule",
      r"\endlastfoot",
    ]
    # ---- table body -------------------------------------------------- #
    for dim in self.get_dims():
      dim_title = " ".join(w.capitalize() for w in dim.split("_"))
      dim_summary = self.dim_summaries_dict.get(dim, "").strip()
      dim_template = self.dim_templates_dict[dim]
      lines += [
        r"\rowcolor{SkyBlue!15}",
        # Fix multicolumn to use exact width of 0.9\textwidth without syntax errors
        r"\multicolumn{3}{p{0.953\textwidth}}{",
        rf"\textbf{{Dimension: {dim_title}}}\newline",
        rf"\textbf{{Template}}: \textit{{{self._escape_latex(dim_template)}}}\newline",
        rf"\textbf{{Summary}}: {self._escape_latex(dim_summary.rsplit('\n', 2)[0]).replace('\n', r'\par ')}",
        r"} \\",
        r"\midrule"
      ]

      # Each sub-dimension row
      for sub_dim, fields in self.dim_data_dict[dim].items():
        sub = self._escape_latex(sub_dim)
        definition = self._escape_latex(fields["definition"])
        example = self._escape_latex(fields["example"])
        lines.append(f"{sub} & {definition} & {example} \\\\")

      lines.append(r"\midrule")

    # ---- end of table ------------------------------------------------- #
    lines += [
        r"\end{longtable}",
        r"\endgroup"
    ]

    latex_lines = "\n".join(lines)
    with open(path, "w") as f:
        f.write(latex_lines)