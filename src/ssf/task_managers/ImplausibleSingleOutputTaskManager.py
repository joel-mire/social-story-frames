from ssf.task_managers.TaskManager import TaskManager
from ssf.prompt_builders.ImplausibleInferenceGenerationPromptBuilder import ImplausibleInferenceGenerationPromptBuilder
from ssf.Constants import *
from ssf.utils.OutputParser import OutputParser

class ImplausibleSingleOutputTaskManager(TaskManager):
    """Task manager for implausible generation tasks"""

    def get_prompt(self, dim, row):
        """Build prompt using ImplausibleInferenceGenerationPromptBuilder"""
        prompt = (
            ImplausibleInferenceGenerationPromptBuilder(taxonomy=self.taxonomy, dim=dim, text=row['_text'], single_output=True)
            .community_name(row[COMMUNITY_META_COL] if self.context.include_community_name else None)
            .community_description(row[COMMUNITY_DESCRIPTION_META_COL] if self.context.include_community_description else None)
            .community_values(row[COMMUNITY_VALUES_META_COL] if self.context.include_community_values else None)
            .progenitor_summary(row[PROGENITOR_SUMMARY_META_COL] if self.context.include_progenitor_summary else None)
            .conversation_summary(row[CONVERSATION_SUMMARY_META_COL] if self.context.include_conversation_summary else None)
            .build()
        )
        return prompt
    
    def add_results(self, stories_df, dim_outputs_dict):
        """Parse single-output results and add to DataFrame using ID-based matching"""
        for dim, dim_outputs in dim_outputs_dict.items():
            # Create ID to output mapping
            id_to_output = {}
            for entry in dim_outputs:
                story_id = entry.get('id')
                if story_id:
                    parsed_output = OutputParser.parse_single_response(entry['output'])
                    id_to_output[story_id] = parsed_output

            free_text_col = f'{dim}_gen'
            col_name = f"{self.disambiguator}${free_text_col}0"

            # Match by ID
            for i, row in stories_df.iterrows():
                story_id = row['id']
                if story_id in id_to_output:
                    stories_df.at[i, col_name] = id_to_output[story_id]

        return stories_df