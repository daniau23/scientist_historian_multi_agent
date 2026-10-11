# Scripting

Runnable Python package and CLI for the 12 workflows.

## Usage
`Bash
python -m scripting list
python -m scripting run prompt_chaining --scientist "Isaac Newton"
python -m scripting run parallelization_feedback_tools --scientist "Marie Curie"
`

## Key files
- cli.py - list/run/convert commands
- llms.py, 	tools.py, 
odes.py, prompts.py, schemas.py - core engine
- 
registry.py, workflows/ - workflow definitions and discovery
- 	tracing.py, pii.py, config.py, models.py - utilities

Engine files mirror deployment/ for parity.
