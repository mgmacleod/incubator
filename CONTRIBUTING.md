# Contributing to Project Incubator

Thank you for your interest in experimenting with new ideas! This guide will help you get started.

## Creating a New Project

### 1. Choose a Descriptive Name

Pick a clear, descriptive name for your project folder:
- Use lowercase with hyphens: `my-cool-project`
- Keep it concise but meaningful
- Avoid generic names like `test` or `experiment-1`

### 2. Set Up Your Project Folder

```bash
mkdir projects/your-project-name
cd projects/your-project-name
```

### 3. Add a README

Every project must have a `README.md` with at least:

```markdown
# Project Name

## Status
[Active | Paused | Archived | Promoted]

## Description
Brief description of what you're trying to accomplish.

## Technologies
- List of technologies/frameworks used

## Goals
- What you're trying to learn or build
- Success criteria (if any)

## Getting Started
How to run/use the project (if applicable).

## Notes
Any important notes or learnings.
```

## Best Practices

### Do's ✅

- **Do** keep your project self-contained in its own folder
- **Do** add a `.gitignore` if your project generates build artifacts
- **Do** document your progress and learnings
- **Do** clean up sensitive data before committing
- **Do** keep dependencies up to date if continuing work
- **Do** use conventional folder structures (src/, tests/, docs/, etc.)

### Don'ts ❌

- **Don't** commit secrets, API keys, or credentials
- **Don't** commit large binary files (use Git LFS if needed)
- **Don't** modify other people's projects without permission
- **Don't** commit generated files or build artifacts
- **Don't** add dependencies to the root of the repository

## Project Status Updates

Update your project's README status as it evolves:

### Active
Currently being worked on. Include recent updates in the README.

### Paused
On hold but may resume. Document where you left off.

### Archived
No longer active. Document:
- Final state
- What worked
- What didn't work
- Key learnings
- Reasons for discontinuation

### Promoted
Moved to a dedicated repository. Document:
- Link to new repository
- Why it was promoted
- Brief summary of what was learned during incubation

## Code Quality

Since this is an experimental space, we don't enforce strict code quality standards. However:

- Write code you won't be embarrassed by
- Add basic comments for complex logic
- Keep it readable for your future self
- If you're testing best practices, follow them!

## Collaboration

- Feel free to explore and learn from other projects
- If you want to contribute to someone else's project, open an issue or PR
- Be respectful and constructive in feedback

## Moving Projects

### Graduating to Production

When your project is ready to move to a dedicated repository:

1. Create a new repository for the project
2. Copy the code to the new repository
3. Set up proper CI/CD, documentation, etc.
4. Update your project README here with:
   ```markdown
   ## Status
   Promoted to [new-repo-link]
   
   This project has graduated from the incubator and now lives at [link].
   Kept here for reference.
   ```

### Archiving

When discontinuing a project:

1. Update the README with final status and learnings
2. Set status to "Archived"
3. Document what you learned (even from failures!)

## Getting Help

- Open an issue for questions
- Reach out to maintainers
- Check other projects for examples

## Have Fun!

This is a space for learning and experimentation. Don't be afraid to try new things, break things, and learn from the process. 🚀
