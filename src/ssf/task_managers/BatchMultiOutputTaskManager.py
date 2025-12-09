import os
import json
import openai
from ssf.Constants import *
from ssf.utils.OutputParser import OutputParser
from ssf.Exceptions import BatchNotReadyError
from ssf.Configs import Context
from ssf.prompt_builders.InferenceGenerationPromptBuilder import InferenceGenerationPromptBuilder

class BatchMultiOutputTaskManager:
    """Non-blocking batch manager using OpenAI Batch API"""
    
    def __init__(self, 
                 taxonomy,
                 model_name: str,
                 out_dir: str,
                 force_redo: bool,
                 disambiguator: str,
                 context_config: Context,
                 poll_interval: int = 10, 
                 completion_window: str = "24h"):
        
        self.taxonomy = taxonomy
        self.model_name = model_name
        self.out_dir = out_dir
        self.force_redo = force_redo
        self.disambiguator = disambiguator
        self.context = context_config
        self.poll_interval = poll_interval
        self.completion_window = completion_window
        
        self.job_dir = os.path.join(self.out_dir, self.disambiguator, 'batch_jobs')
        os.makedirs(self.job_dir, exist_ok=True)
        self.jobs_file = os.path.join(self.job_dir, f"batch_job_{self.disambiguator}.json")
        
        print(f"[BatchManager] Initialized, job metadata at {self.jobs_file}")

    def add_results(self, stories_df, dim_outputs_dict):
        """Add multi-output results to DataFrame"""
        for dim, dim_outputs in dim_outputs_dict.items():
            id_to_output = {}
            for entry in dim_outputs:
                story_id = entry['id'].split("||")[0]
                free_text = OutputParser.parse_multi_response(entry['output'])
                id_to_output[story_id] = free_text
            
            free_text_col = f'{dim}_gen'
            prefix = self.disambiguator.replace("-", "_")
            
            for i, row in stories_df.iterrows():
                story_id = row['id']
                if story_id in id_to_output:
                    texts = id_to_output[story_id]
                    stories_df.at[i, f"{prefix}_{free_text_col}0"] = texts[0] if len(texts) > 0 else None
                    stories_df.at[i, f"{prefix}_{free_text_col}1"] = texts[1] if len(texts) > 1 else None
                    stories_df.at[i, f"{prefix}_{free_text_col}2"] = texts[2] if len(texts) > 2 else None
        
        return stories_df
    
    def run_task(self, stories_df):
        """Execute batch task - returns DataFrame when complete, raises BatchNotReadyError if not ready"""
        batch_jsonl = os.path.join(self.out_dir, 'outputs', self.disambiguator, 'batch.jsonl')
        if os.path.exists(batch_jsonl):
            print("[BatchManager] Reading existing batch.jsonl...")
            dim_outputs_dict = {}
            with open(batch_jsonl, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    dim = entry['id'].split("||")[-1]
                    if dim not in dim_outputs_dict:
                        dim_outputs_dict[dim] = []
                    dim_outputs_dict[dim].append(entry)
            print("[BatchManager] Calling add_results...")
            stories_df = self.add_results(stories_df, dim_outputs_dict)
            print("🎉 Batch complete—DataFrame has all generated columns.")
            return stories_df

        print("[BatchManager] No batch.jsonl found. Proceeding with regular task flow...")

        prompts_path = os.path.join(self.out_dir, 'prompts', self.disambiguator, 'batch_tasks.jsonl')
        os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
        outputs_path = os.path.join(self.out_dir, 'outputs', self.disambiguator)
        os.makedirs(outputs_path, exist_ok=True)

        if not os.path.exists(self.jobs_file) or self.force_redo:
            print("[BatchManager] Preparing batch tasks...")
            tasks = []
            prompt_ids = []

            for dim in self.taxonomy.get_dims():
                for idx, row in stories_df.iterrows():
                    id = row['id']
                    custom_id = f"{id}||{dim}"

                    # Build prompt using context config
                    prompt = (
                        InferenceGenerationPromptBuilder(taxonomy=self.taxonomy, dim=dim, text=row['_text'], single_output=False)
                        .community_name(row[COMMUNITY_META_COL] if self.context.include_community_name else None)
                        .community_description(row[COMMUNITY_DESCRIPTION_META_COL] if self.context.include_community_description else None)
                        .community_values(row[COMMUNITY_VALUES_META_COL] if self.context.include_community_values else None)
                        .progenitor_summary(row[PROGENITOR_SUMMARY_META_COL] if self.context.include_progenitor_summary else None)
                        .conversation_summary(row[CONVERSATION_SUMMARY_META_COL] if self.context.include_conversation_summary else None)
                        .build()
                    )

                    body = {
                        "model": self.model_name,
                        "max_tokens": 1000,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    tasks.append({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body
                    })
                    prompt_ids.append(custom_id)

            with open(prompts_path, 'w') as f:
                for t in tasks:
                    f.write(json.dumps(t) + "\n")
            print(f"[BatchManager] Wrote {len(tasks)} tasks to {prompts_path}")

            batch_file = openai.files.create(file=open(prompts_path, 'rb'), purpose='batch')
            print(f"[BatchManager] Uploaded batch file id={batch_file.id}")

            batch_job = openai.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=self.completion_window
            )
            print(f"[BatchManager] Created batch job id={batch_job.id}")

            with open(self.jobs_file, 'w') as jf:
                json.dump({
                    'job_id': batch_job.id,
                    'prompt_ids': prompt_ids,
                    'outputs_path': outputs_path
                }, jf)
            print(f"[BatchManager] Recorded metadata {self.jobs_file}")
            print("✅ Batch job submitted. Re-run this line later to poll status.")
            raise BatchNotReadyError()

        print("[BatchManager] Checking batch status...")
        with open(self.jobs_file, 'r') as jf:
            meta = json.load(jf)

        job = openai.batches.retrieve(meta['job_id'])
        print(f"[BatchManager] Status={job.status}")

        if job.status in ['failed', 'cancelled']:
            print("[BatchManager] Batch failed or cancelled. Starting fresh.")
            os.remove(self.jobs_file)
            return self.run_task(stories_df)

        if job.status != 'completed':
            print("⏳ Batch still running—re-run later to check again.")
            raise BatchNotReadyError()

        result_file = openai.files.content(job.output_file_id)
        lines = result_file.text.splitlines()
        print(f"[BatchManager] Downloaded {len(lines)} lines")

        choice_map = {}
        for ln in lines:
            entry = json.loads(ln)
            choice_map[entry['custom_id']] = entry['response']['body']['choices'][0]['message']['content']

        prompt_ids = meta['prompt_ids']
        outputs_path = meta['outputs_path']

        for dim in self.taxonomy.get_dims():
            dim_output_path = os.path.join(outputs_path, f"{dim}.jsonl")
            with open(dim_output_path, 'w') as outf:
                for cid in prompt_ids:
                    if f"||{dim}" in cid:
                        outf.write(json.dumps({'id': cid, 'output': choice_map.get(cid)}) + "\n")

        print(f"[BatchManager] Wrote dimension-specific results to {outputs_path}")

        # Load and organize results by dimension
        dim_outputs_dict = {}
        for dim in self.taxonomy.get_dims():
            dim_output_path = os.path.join(outputs_path, f"{dim}.jsonl")
            with open(dim_output_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    dim = entry['id'].split("||")[-1]
                    if dim not in dim_outputs_dict:
                        dim_outputs_dict[dim] = []
                    dim_outputs_dict[dim].append(entry)

        print("[BatchManager] Calling add_results...")
        result_df = self.add_results(stories_df, dim_outputs_dict)
        print("🎉 Batch complete—DataFrame has all generated columns.")
        return result_df