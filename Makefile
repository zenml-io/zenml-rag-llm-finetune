stack_name ?= rag_finetune_stack
setup:
	pip install -r requirements.txt
	zenml integration install openai pillow -y

install-stack:
	@echo "Specify stack name [$(stack_name)]: " && read input && [ -n "$$input" ] && stack_name="$$input" || stack_name="$(stack_name)" && \
	zenml stack register -a default -o default $${stack_name} && \
	zenml stack set $${stack_name} && \
	zenml stack up
