"""
Generic OpenAI Batch API processor for handling batch job submission, monitoring, and results.
Separates batch management from prompt generation logic.
"""
import os
import json
import openai

class BatchNotReadyError(Exception):
    """Raised when batch job is not yet completed."""
    pass

class BatchProcessor:
    """
    Generic batch processor for OpenAI Batch API.
    Handles submission, monitoring, and result downloading.
    """
    
    def __init__(self, model_name, completion_window="24h", show_progress=False):
        self.model_name = model_name
        self.completion_window = completion_window
        self.show_progress = show_progress
    
    def process_batch(self, prompts, custom_ids, output_path, force_redo=False):
        """
        Process a batch of prompts using OpenAI Batch API.
        
        Args:
            prompts: List of prompt strings
            custom_ids: List of custom IDs (same length as prompts)
            output_path: Path to save results
            force_redo: Whether to regenerate if output exists
            
        Returns:
            Path to output file
            
        Raises:
            BatchNotReadyError: When batch job is submitted but not yet complete
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        job_metadata_path = self._get_job_metadata_path(output_path)
        
        # Check for existing batch job first (priority over cached results)
        if os.path.exists(job_metadata_path) and not force_redo:
            job = self._try_get_completed_job(job_metadata_path)
            if job:
                self._download_and_save_results(job, output_path)
                return output_path
        
        # Check for cached results only if no batch job exists
        if not force_redo and os.path.exists(output_path) and not os.path.exists(job_metadata_path):
            if self.show_progress:
                print(f"[BatchProcessor] Using cached results from {output_path}")
            return output_path
        
        # Create and submit new batch job
        tasks = self._create_batch_tasks(prompts, custom_ids)
        
        if not tasks:
            # No valid tasks, create empty output file
            open(output_path, 'w').close()
            return output_path
        
        # Check if batch exceeds limits and split if necessary
        MAX_BATCH_SIZE = 50000
        if len(tasks) > MAX_BATCH_SIZE:
            if self.show_progress:
                print(f"[BatchProcessor] Batch size {len(tasks)} exceeds limit of {MAX_BATCH_SIZE}, splitting into multiple batches...")
            self._submit_split_batch_jobs(tasks, output_path, job_metadata_path)
        else:
            batch_requests_path = output_path.replace('_outputs.jsonl', '_batch_requests.jsonl')
            self._submit_batch_job(tasks, batch_requests_path, job_metadata_path)
        
        raise BatchNotReadyError("Batch job submitted, not yet completed")
    
    def _get_job_metadata_path(self, output_path):
        """Get path for batch job metadata."""
        output_dir = os.path.dirname(output_path)
        job_dir = os.path.join(output_dir, 'batch_jobs')
        os.makedirs(job_dir, exist_ok=True)
        return os.path.join(job_dir, 'batch_job.json')
    
    def _create_batch_tasks(self, prompts, custom_ids):
        """Create batch API tasks from prompts and custom IDs."""
        tasks = []
        for prompt, custom_id in zip(prompts, custom_ids):
            if not prompt or prompt.strip() == '':
                continue  # Skip empty prompts
                
            body = {
                "model": self.model_name,
                "max_tokens": 200,
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
        return tasks
    
    def _submit_batch_job(self, tasks, batch_requests_path, job_metadata_path):
        """Submit batch job to OpenAI and save metadata."""
        # Save batch requests
        with open(batch_requests_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        
        if self.show_progress:
            print(f"[BatchProcessor] Uploading {len(tasks)} batch requests...")
            
        # Upload and create batch
        batch_file = openai.files.create(
            file=open(batch_requests_path, 'rb'),
            purpose='batch'
        )
        
        batch_job = openai.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window
        )
        
        # Save job metadata
        with open(job_metadata_path, 'w') as f:
            json.dump({
                'job_id': batch_job.id,
                'file_id': batch_file.id,
                'status': 'submitted'
            }, f)
            
        if self.show_progress:
            print(f"[BatchProcessor] Batch job submitted: {batch_job.id}")
    
    def _try_get_completed_job(self, job_metadata_path):
        """Try to get completed batch job, return None if not ready."""
        try:
            with open(job_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Handle split batches
            if metadata.get('split_batch', False):
                return self._check_split_batch_status(job_metadata_path)
            else:
                return self._check_batch_status(job_metadata_path)
        except BatchNotReadyError:
            raise  # Re-raise to caller
    
    def _check_batch_status(self, job_metadata_path):
        """Check status of existing batch job."""
        with open(job_metadata_path, 'r') as f:
            metadata = json.load(f)
            
        job = openai.batches.retrieve(metadata['job_id'])
        
        if self.show_progress:
            print(f"[BatchProcessor] Batch status: {job.status}")
            
        if job.status in ['failed', 'cancelled']:
            if self.show_progress:
                print(f"[BatchProcessor] Batch {job.status}, cleaning up...")
            os.remove(job_metadata_path)
            raise BatchNotReadyError(f"Batch job {job.status}. Please run again to restart.")
        elif job.status == 'completed':
            return job
        else:
            raise BatchNotReadyError(f"Batch job not completed yet. Status: {job.status}")
    
    def _download_and_save_results(self, job, output_path):
        """Download batch results and save to output file."""
        # Download results
        result_file = openai.files.content(job.output_file_id)
        lines = result_file.text.splitlines()
        
        if self.show_progress:
            print(f"[BatchProcessor] Downloaded {len(lines)} results")
            
        # Parse and save results
        results = []
        for line in lines:
            entry = json.loads(line)
            custom_id = entry['custom_id']
            content = entry['response']['body']['choices'][0]['message']['content']
            results.append({
                'custom_id': custom_id,
                'output': content
            })
            
        # Save results
        with open(output_path, 'w') as output_file:
            for result in results:
                output_file.write(json.dumps(result) + "\n")
    
    def _submit_split_batch_jobs(self, tasks, output_path, job_metadata_path):
        """Submit multiple batch jobs for large batches that exceed limits."""
        MAX_BATCH_SIZE = 50000
        
        # Split tasks into chunks
        task_chunks = []
        for i in range(0, len(tasks), MAX_BATCH_SIZE):
            task_chunks.append(tasks[i:i + MAX_BATCH_SIZE])
        
        if self.show_progress:
            print(f"[BatchProcessor] Splitting {len(tasks)} tasks into {len(task_chunks)} batches")
        
        # Submit each chunk as a separate batch job
        job_ids = []
        for chunk_idx, chunk in enumerate(task_chunks):
            # Create unique paths for each chunk
            chunk_requests_path = output_path.replace('_outputs.jsonl', f'_batch_requests_chunk_{chunk_idx}.jsonl')
            chunk_metadata_path = job_metadata_path.replace('batch_job.json', f'batch_job_chunk_{chunk_idx}.json')
            
            # Submit chunk
            self._submit_batch_job(chunk, chunk_requests_path, chunk_metadata_path)
            
            # Track job metadata
            with open(chunk_metadata_path, 'r') as f:
                metadata = json.load(f)
                job_ids.append(metadata['job_id'])
        
        # Save master metadata with all job IDs
        with open(job_metadata_path, 'w') as f:
            json.dump({
                'split_batch': True,
                'job_ids': job_ids,
                'num_chunks': len(task_chunks),
                'status': 'submitted'
            }, f)
        
        if self.show_progress:
            print(f"[BatchProcessor] Submitted {len(task_chunks)} batch jobs: {job_ids}")