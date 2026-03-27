RUN_ID ?= $(shell date +%Y-%m-%d-%H%M)
PARALLELISM ?= 40
BASELINE ?=
NAMESPACE ?= torch-spyre-cicd

.PHONY: discover status logs fetch diff clean list dry-run help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

discover: ## Launch a full discovery run
	python -m discovery launch-orchestrator \
		--run-id $(RUN_ID) \
		--parallelism $(PARALLELISM) \
		--namespace $(NAMESPACE) \
		$(if $(BASELINE),--baseline $(BASELINE),)
	@echo ""
	@echo "Orchestrator launched. Use 'make status' or 'make logs' to monitor."

dry-run: ## Generate all YAML without applying
	python -m discovery launch-orchestrator \
		--run-id $(RUN_ID) \
		--parallelism $(PARALLELISM) \
		--namespace $(NAMESPACE) \
		--dry-run

status: ## Check progress of running discovery
	python -m discovery status --run-id $(RUN_ID) --namespace $(NAMESPACE)

logs: ## Tail orchestrator logs
	python -m discovery logs --run-id $(RUN_ID) --namespace $(NAMESPACE) --follow

fetch: ## Fetch results to local machine
	python -m discovery fetch-results --run-id $(RUN_ID) --namespace $(NAMESPACE)

diff: ## Diff against baseline (requires BASELINE=<run-id>)
	@test -n "$(BASELINE)" || (echo "Usage: make diff BASELINE=<run-id>" && exit 1)
	python -m discovery diff --run-id $(RUN_ID) --baseline $(BASELINE)

clean: ## Delete K8s resources for a run
	python -m discovery clean --run-id $(RUN_ID) --namespace $(NAMESPACE)

list: ## List all discovery runs
	python -m discovery list-runs --namespace $(NAMESPACE)

resume: ## Resume a failed/interrupted run
	python -m discovery resume --run-id $(RUN_ID) --namespace $(NAMESPACE)
