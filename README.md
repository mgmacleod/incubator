# Project Incubator 🚀

A collaborative space to start and explore early phases of experimental projects.

## Purpose

This repository serves as an incubator for experimental ideas and proof-of-concept projects. It's a low-friction environment where you can:
- Start new experimental projects without the overhead of creating a new repository
- Try out new technologies, frameworks, or approaches
- Collaborate on early-stage ideas
- Test concepts before committing to a full project

## How It Works

1. **Start a Project**: Create a new folder in the `projects/` directory with your project name
2. **Experiment**: Develop your idea, iterate, and learn
3. **Decide**: After experimentation, decide on the project's future:
   - **Promote**: Move successful projects to their own dedicated repository
   - **Archive**: Keep unsuccessful experiments here as learning references

## Quick Start

### Creating a New Project

1. Create a new folder in the `projects/` directory:
   ```bash
   mkdir projects/your-project-name
   cd projects/your-project-name
   ```

2. Add a `README.md` describing your project:
   - What problem are you exploring?
   - What technologies are you using?
   - Current status and goals

3. Start building!

### Project Structure

Each project should be self-contained in its own folder:

```
projects/
├── your-project-name/
│   ├── README.md          # Project description and status
│   ├── src/               # Source code
│   ├── tests/             # Tests (if applicable)
│   └── ...                # Other project files
```

## Guidelines

- **Keep it simple**: Don't overthink the structure - just start coding
- **Document as you go**: A simple README is sufficient
- **Be respectful**: Clean up after yourself, don't commit secrets or large binaries
- **Stay independent**: Each project should be self-contained
- **Update status**: Keep your project README updated with current status

## Project Lifecycle

### Active Development
Projects in active development should have a clear README with goals and current status.

### Archiving
When a project is no longer active, update its README with:
- Final status
- Lessons learned
- Why it was discontinued (if applicable)

### Promoting to Production
When a project graduates:
1. Create a new dedicated repository
2. Move the code to the new repository
3. Update the project README here with a link to the new location
4. Keep the folder as a reference

## Examples

Check out the `projects/` directory for examples and templates.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Questions?

Open an issue or reach out to the repository maintainers.
