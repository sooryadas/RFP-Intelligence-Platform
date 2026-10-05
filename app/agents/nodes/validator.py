from langsmith import traceable


@traceable(name="Validator Node")
def validator_node(state: dict) -> dict:
    """Check that extracted values have citations."""
    extracted_data = state.get("extracted_data", {})
    failed_fields = []

    for field_name, field in extracted_data.items():
        value = field.get("value")
        sources = field.get("sources", [])

        if value is not None and not sources:
            failed_fields.append(field_name)

    validation_result = {
        "passed": len(extracted_data) - len(failed_fields),
        "failed": len(failed_fields),
        "failed_fields": failed_fields,
    }

    return {
        "validation_result": validation_result,
        "retry_count": state.get("retry_count", 0)
        + (1 if failed_fields else 0),
    }