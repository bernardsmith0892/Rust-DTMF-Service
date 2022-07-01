use std::{io::{self, Write}, sync::Mutex};
use cpal::{traits::{HostTrait, StreamTrait}};
use rodio::DeviceTrait;
use chrono;

use crate::controller::DtmfCommand;

pub mod dtmf;
pub mod controller;

fn process_input_stream(data: &[f32], processor: &mut dtmf::DtmfProcessor, commands: &Vec<DtmfCommand>, current_input: &Mutex<String>) {
    let mut input = current_input.lock().unwrap();

    match processor.process_samples(data) {
        // Attempt to current input to a command if a '#' was provided
        Some('#') => {
            print!("{}", "#\n");
            io::stdout().flush().unwrap();

            for cmd in commands {
                if cmd.validate_passcode(&input) {
                    match cmd.run() {
                        Ok(output) => {
                            match std::str::from_utf8(&output.stdout) {
                                Ok(utf8_output) => println!("Output: {}", utf8_output.trim()),
                                Err(error) => println!("Issue decoding command output to UTF-8! ({:?})", error),
                            };
                        },
                        Err(error) => println!("Error running command {:?} - {:?}", cmd, error),
                    };
                }
            }

            input.clear();
        },
        // Clear input string if we decode a space character
        Some(' ') => {
            // Add a newline if we're currently decoding a command
            if !input.is_empty() {
                print!("{}", "\n");
                io::stdout().flush().unwrap();
            }

            input.clear();
        },
        // Add all other characters to the current input string
        Some(char) => {
            if input.is_empty() {
                let curr_time = chrono::Local::now();
                print!("{} - ", curr_time.format("%d-%b-%Y %H:%M:%S%.3f"))
            }

            print!("{}", char);
            io::stdout().flush().unwrap();

            input.push(char);
        },
        None => (),
    }
}

fn main() {
    // let sine_hz = |n: i32, freq: f32, samp_rate: f32| 
      // f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate);

    // let sample_rate: u32 = 48_000; 
    // let test_signal: Vec<f32> = (0..480)
        // .map(|n| sine_hz(n-16, 1000.0, sample_rate as f32))
        // .collect();
    
    // // Test magnitude of two frequencies
    // for freq in (500..1500).step_by(100) {
        // let g_power = dtmf::goertzel(freq as f32, sample_rate, &test_signal);
        // let analytic_sine: Vec<f32> = (0..440).map(|n| sine_hz(n, freq as f32, sample_rate as f32)).collect();
        // //let analytic_cosine: Vec<f32> = (0..124).map(|n| cosine_hz(n, freq1 as f32, sample_rate as f32) + cosine_hz(n, freq2 as f32, sample_rate as f32)).collect();
        // let c_sine = dtmf::cross_correlate(&test_signal, &analytic_sine);//.into_iter().reduce(f32::max).unwrap();
        // let c_h = dtmf::hilbert(&c_sine, sample_rate).into_iter().reduce(f32::max).unwrap();
        // //let c_cosine: f32 = dtmf::cross_correlate(&test_signal, &analytic_cosine).into_iter().reduce(f32::max).unwrap().abs();
        // println!("{} Hz, {:.3}, {:.3?}", freq, g_power, c_h);
    // }

    let commands = DtmfCommand::from_file("dtmf_config.txt");
    let current_input: Mutex<String> = Mutex::new(String::new());

    let host = cpal::default_host();
    let devices = host.input_devices().expect("no input devices available!");
    println!("Select input device:");
    for (i, device) in devices.enumerate() {
        println!("\t{} - {:?}", i, device.name().unwrap());
    }

    let mut input_string = String::new();
    std::io::stdin().read_line(&mut input_string).unwrap();
    let device_index = input_string.trim().parse::<i32>().expect("must provide a number!");


    let device = host.input_devices().unwrap().nth(device_index as usize).expect("cannot use this device!");
    let config = device.default_input_config().unwrap().config();

    let mut processor = dtmf::DtmfProcessor::new(config.sample_rate.0, config.channels, dtmf::Mode::Goertzel); 

    let stream = device.build_input_stream(
        &config, 
        move |data, _| process_input_stream(data, &mut processor, &commands, &current_input),
        |_| {}
    ).expect("cannot build stream!");

    stream.play().unwrap();


    loop { }
}