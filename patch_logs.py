import os
import glob
import re

files = glob.glob("**/*.py", recursive=True)
for filepath in files:
    if '.venv' in filepath or 'scratch' in filepath:
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if we need to add logging setup
    needs_logger = 'logger.info(' in content or 'logger.info(' in content
    if not needs_logger:
        continue
        
    # Replace logger.info(  with logger.info(
    content = re.sub(r'\bprint\s*\(', 'logger.info(', content)
    
    # Ensure import logging and logger exists
    if 'import logging' not in content:
        # insert after the first import or at top
        content = "import logging\nlogger = logging.getLogger(__name__)\n" + content
    elif 'logger = logging.getLogger(' not in content:
        # Add logger = logging.getLogger...
        # find import logging
        content = re.sub(r'import logging', "import logging\nlogger = logging.getLogger(__name__)", content, count=1)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    logger.info(f"Patched {filepath}")

