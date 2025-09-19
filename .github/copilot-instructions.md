# GitHub Copilot Instructions

## Code Quality & Standards
- Use proper Python standards and PEP 8 compliance
- Use type hints with `from __future__ import annotations`
- Run mypy, ruff, black, and isort for all code
- No unused imports or variables
- All imports should be at the top of files, organized with isort
- Follow single responsibility principle

## Dependencies & Tools
- Don't reinvent the wheel - use existing libraries and tools
- If dependencies are declared in pyproject.toml, import them directly (no conditional imports)
- Use uv for dependency management
- Use established libraries instead of custom implementations
- Prefer librosa/soundfile for audio processing over manual numpy operations
- Use proper audio processing libraries instead of custom converters
- Use uv, poetry commands
- Use pydantic and pydantic-settings

## File Organization
- Only create files in appropriate project directories (src/, tests/, examples/, docs/)
- Follow existing project structure
- Don't create files in project root unless absolutely necessary
- Ask before creating files outside standard structure

## Error Handling
- Provide specific, actionable error messages
- Use proper exception types with context
- Include error codes and details in exceptions
- Implement graceful fallbacks where appropriate

## Code Simplification
- Keep classes focused on single responsibility
- Avoid complex conditional logic when dependencies are guaranteed
- Remove unnecessary complexity and abstractions
- Make code readable and maintainable
- Use standard patterns and practices

## Testing
- Ensure all functionality is properly tested
- Run tests after making changes
- Use pytest for testing framework
- Write comprehensive test coverage

## Documentation
- Use clear, concise docstrings
- Document public APIs properly
- Keep comments focused on "why" not "what"

## Performance
- Use appropriate libraries for performance-critical operations
- Implement proper device detection and fallbacks
- Consider memory usage and efficiency
- Use proper async patterns where needed

## Pain Points to Avoid
- No scattered imports throughout methods
- No conditional imports when dependencies are declared
- No reinventing standard functionality
- No overly complex classes with multiple responsibilities
- No creating files outside project structure
- No unused complexity or over-engineering
