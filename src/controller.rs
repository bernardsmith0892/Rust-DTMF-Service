use std::fs::File;
use std::io::{BufReader, self, BufRead};
use std::process::{Command, Output};
use shlex;
use regex::Regex;

#[derive(Debug)]
pub struct DtmfCommand {
    pub passcode: String,
    pub program: String,
    pub args: Vec<String>,
}

impl DtmfCommand {
    /// Generates a `DtmfCommand` from a comma-separated string in the format of:
    /// `<numeric password>,<POSIX-compliant shell command>
    pub fn new(line: &str) -> Option<Self> {
        // Split the string in two on the first comma
        let mut comma_split = line.splitn(2, ',');

        // The code should only contain numbers and letters A through D
        let passcode = comma_split.next()?;
        let re = Regex::new(r"^[0-9a-dA-D]+$").unwrap();
        if !re.is_match(passcode) {
            return None;
        }

        let command_string = comma_split.next()?;
        let posix_command = shlex::split(command_string)?;
        let split_command = posix_command.split_first()?;

        Some(
            Self {
                passcode: passcode.trim().to_string(),
                program: split_command.0.to_string(),
                args: split_command.1.iter()
                    .map(|s| s.to_string())
                    .collect(),
            }
        )
    }

    pub fn validate_passcode(&self, passcode: &str) -> bool {
        self.passcode == passcode
    }

    pub fn run(&self) -> io::Result<Output> {
        Command::new(&self.program)
            .args(&self.args)
            .output()
    }

    pub fn from_file(filename: &str) -> Vec<Self> {
        let file = BufReader::new(File::open(filename).unwrap());
        let mut commands = Vec::<Self>::new();
        for (line_num, line) in file.lines().enumerate() {
            if let Ok(command_line) = line {
                match Self::new(&command_line) {
                    Some(new_command) => commands.push(new_command),
                    None => println!("Error decoding configuration on line {} - {:?}", line_num + 1, command_line),
                }
            }
        }

        commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_dtmf_command() {
        let command_line = "1234567890ABCD, cmd.exe /c echo hello world!";
        let test_command = DtmfCommand::new(command_line).unwrap();

        // Validate passcode parsing and passcode validation
        assert_eq!(test_command.passcode, "1234567890ABCD");
        assert!(test_command.validate_passcode("1234567890ABCD"));

        // Validate command parsing
        assert_eq!(test_command.program, "cmd.exe");
        assert_eq!(test_command.args, ["/c", "echo", "hello", "world!"]);

        // Validate command execution and output
        let command_output = test_command.run().unwrap().stdout;
        let output_str = std::str::from_utf8(&command_output).unwrap().trim();
        assert_eq!(output_str, "hello world!");
    }

    #[test]
    fn invalid_dtmf_command() {
        // No shell command
        let command_line = "123456789, ";
        let test_command = DtmfCommand::new(command_line);
        assert!(test_command.is_none(), "Invalid command '{}' appears valid! {:#?}", command_line, test_command);

        // Shell command with unescaped quotes
        let command_line = "123456789, echo \"Hi!";
        let test_command = DtmfCommand::new(command_line);
        assert!(test_command.is_none(), "Invalid command '{}' appears valid! {:#?}", command_line, test_command);

        // Invalid Code
        let command_line = "12EFG, echo hello!";
        let test_command = DtmfCommand::new(command_line);
        assert!(test_command.is_none(), "Invalid command '{}' appears valid! {:#?}", command_line, test_command);
    }
}