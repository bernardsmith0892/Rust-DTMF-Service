use std::{f32::consts::PI, sync::Mutex};
use cpal::{StreamInstant};

const DTMF_FREQUENCIES: [u32; 8] = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
const MIN_THRESHOLD: f32 = 2.0;

pub struct DtmfProcessor {
    pub sample_rate: u32,
    pub channels: u16,
    input_buffer: Mutex<Vec<f32>>,
    minimum_n: u32,
    candidate: (char, Option<StreamInstant>, bool),
}

pub const TONE_LENGTH: u128 = 20; // ms
pub const SPACE_LENGTH: u128 = 500; // ms

impl DtmfProcessor {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self { 
            sample_rate: sample_rate,
            channels: channels, 
            input_buffer: Mutex::new(Vec::new()), 
            minimum_n: sample_rate / 70, // Buffer must have enough samples to allow under 70Hz wide bins
            candidate: (' ', None, true),
        }
    }

    pub fn process_samples(&mut self, data: &[f32], info: &cpal::InputCallbackInfo) -> Option<char> {
        // Unlock the input_buffer and add the current data
        let mut input_buffer = self.input_buffer.lock().unwrap();
        input_buffer.extend::<Vec<f32>>(
            // Convert multi-channel inputs into mono through averaging
            data.chunks(self.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32 )
            .collect()
        );

        // Initialize the starting timestamp, if needed
        if let (_, None, _) = self.candidate {
            self.candidate = (' ', Some(info.timestamp().capture), true);
        }


        // Process the current input if we have enough samples
        if input_buffer.len() >= self.minimum_n as usize {
            // Analyze the input to detect a DTMF signal
            let current_tone = detect_dtmf_digit(&input_buffer, self.sample_rate);
            let current_time = info.timestamp().capture;

            match current_tone {
                // If we detected a DTMF signal...
                Some(tone) => match self.candidate {
                    // Return this digit if:
                    //  - It matches the known candidate
                    //  - It's been on longer than the minimum tone spacing
                    //  - And it's not already pushed to the output buffer
                    (candidate_tone, Some(start_time), pushed) if tone == candidate_tone => {
                        if current_time.duration_since(&start_time)
                            .unwrap()
                            .as_millis() >= TONE_LENGTH 
                        && !pushed {
                            // Mark this character as pushed
                            self.candidate = (tone, Some(start_time), true);

                            // Print the current char and clear the input buffer
                            input_buffer.clear();
                            return Some(tone);
                        }
                    },
                    // If it's a new digit, set it as the new candidate 
                    _ => self.candidate = (tone, Some(current_time), false),
                },
                // If we do not detect a DTMF signal...
                None => {
                    match self.candidate {
                        // Mark a space if there has been enough silence after a previous digit
                        (' ', Some(silence_start), pushed) => {
                            if current_time.duration_since(&silence_start)
                                .unwrap().as_millis() >= SPACE_LENGTH 
                            && !pushed {
                                // Mark this space as pushed
                                self.candidate = (' ', Some(silence_start), true);     

                                // Print a space and clear the input buffer
                                input_buffer.clear();
                                return Some(' ');
                            }
                        },
                        // Start marking silence if he haven't already
                        _ => self.candidate = (' ', Some(current_time), false),
                    }
                },
            }

            // Clear the input buffer
            input_buffer.clear();
        }

        // No new DTMF character detected
        None
    }
}

/// Returns the power at a target frequency within a signal.
/// 
/// # Arguments
/// 
/// * `target_freq: f32` - The target frequency in *Hertz*.
/// * `sample_rate: u32` - The sample rate of the signal in *samples per second*.
/// * `samples: &[f32]` - The signal to compute on.
/// 
/// # Example
/// 
/// ```
/// # use dtmf;
/// # fn sin(n: u32, freq: f32, samp_rate: f32) -> f32 {
/// #   f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate)
/// # }
/// 
/// // Create 0.5 second, 1000 Hz sine wave at 1,024 samples/sec
/// let test_signal: Vec<f32> = (0..512)
///     .map(|n| sin(n, 1000.0, 1024.0))
///     .collect();
/// 
/// // Test signal strength at two frequencies (1,000 Hz and 900 Hz)
/// let in_band = dtmf::goertzel(1000.0, 1024, &test_signal);
/// let out_of_band = dtmf::goertzel(900.0, 1024, &test_signal);
/// 
/// // The in-band frequency is much larger than the out-of-band one
/// println!("{} > {}", in_band, out_of_band); // 255.99956 > 0.00097311544
/// assert!(in_band > out_of_band);
/// ```
pub fn goertzel(target_freq: f32, sample_rate: u32, samples: &[f32]) -> f32 {
    let omega = (2.0 * PI * target_freq) / sample_rate as f32;
    let coefficient = 2.0 * f32::cos(omega);
    
    // First Stage - Compute the IIR
    let mut q = (0.0, 0.0, 0.0);
    for &sample in samples {
        // Hamming Window
        // let adjusted_sample = sample * (0.5 - 0.25 * f32::cos(2.0 * PI * (n / samples.len()) as f32));
        q.0 = sample + coefficient * q.1 - q.2;
        q.2 = q.1;
        q.1 = q.0;
    }

    // Second Stage - Compute the FIR
    let real = q.1 - q.2 * f32::cos(omega);
    let imaginary = q.2 * f32::sin(omega);

    ( real.powf(2.0) + imaginary.powf(2.0) ).sqrt()
}

fn standard_deviation(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        None
    }
    else {
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 = data.iter().map(|value| (mean - value).powf(2.0)).sum::<f32>() / data.len() as f32;

        Some(variance.sqrt())
    }
}

/// Returns a vector of matching DTMF frequencies for this signal. A frequency is considered matching if its power is higher than the minimum threshold value (2.0) and it is at least one standard deviation stronger than the average power of all eight DTMF frequencies.
/// 
/// # Arguments
/// 
/// * `samples: &[f32]` - The signal to analyze.
/// * `sample_rate: u32` - The sample rate of the signal in *samples per second*.
/// 
/// # Example
/// 
/// ```
/// # use dtmf;
/// # fn sin(n: u32, freq: f32, samp_rate: f32) -> f32 {
/// #   f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate)
/// # }
/// 
/// // Create 0.5 second signal with 697 Hz and 1633 Hz components (DMTF `A`)
/// let test_signal: Vec<f32> = (0..512)
///     .map(|n| sin(n, 697.0, 1024.0) + sin(n, 1633.0, 1024.0))
///     .collect();
/// 
/// let components = dtmf::get_dtmf_components(&test_signal, 1024);
/// println!("{:?}", components); // [697, 1633]
/// assert_eq!((components[0], components[1]), (697, 1633));
/// ```
pub fn get_dtmf_components(samples: &[f32], sample_rate: u32) -> Vec<u32> {
    // Compute the power at each DTMF frequency
    let results: Vec<f32> = 
        DTMF_FREQUENCIES.iter()
        .map(|freq| goertzel(*freq as f32, sample_rate, samples))
        .collect();

    // Determine the average power and standard deviation
    let mean = results.iter().sum::<f32>() / results.len() as f32;
    let std_dev: f32 = standard_deviation(&results).unwrap();

    // Return all frequencies which are stronger than the minimum threshold and are
    //  at least one standard deviation stronger than the average power level.
    results.into_iter()
        .zip(DTMF_FREQUENCIES)
        .filter(|(output, _)| *output >= mean + std_dev && *output >= MIN_THRESHOLD)
        .map(|(_, freq)| freq)
        .collect()
}

/// Attempts to decode a DTMF signal and return its corresponding digit. Returns `None` if it cannot detect a DTMF signal.
/// 
/// # Arguments
/// 
/// * `samples: &[f32]` - The signal to analyze.
/// * `sample_rate: u32` - The sample rate of the signal in *samples per second*.
/// 
/// # Example
/// 
/// ```
/// # use dtmf;
/// # fn sin(n: u32, freq: f32, samp_rate: f32) -> f32 {
/// #   f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate)
/// # }
/// 
/// // Create 0.5 second signal with 697 Hz and 1633 Hz components (DMTF `A`)
/// let a_signal: Vec<f32> = (0..512)
///     .map(|n| sin(n, 697.0, 1024.0) + sin(n, 1633.0, 1024.0))
///     .collect();
///
/// // Create 0.5 second signal with 1000 Hz and 600 Hz components (invalid DTMF)
/// let bad_signal: Vec<f32> = (0..512)
///     .map(|n| sin(n, 1000.0, 1024.0) + sin(n, 600.0, 1024.0))
///     .collect();
/// 
/// let digit = dtmf::detect_dtmf_digit(&a_signal, 1024);
/// println!("{:?}", digit); // Some("A")
/// assert_eq!(digit, Some('A'));
///
/// let digit = dtmf::detect_dtmf_digit(&bad_signal, 1024);
/// println!("{:?}", digit); // None
/// assert_eq!(digit, None);
/// ```
pub fn detect_dtmf_digit(samples: &[f32], sample_rate: u32) -> Option<char> {
    let freqs = get_dtmf_components(samples, sample_rate);
    
    if freqs.len() == 2 {
        match (freqs[0], freqs[1]) {
            (697, 1209) => Some('1'),
            (697, 1336) => Some('2'),
            (697, 1477) => Some('3'),
            (697, 1633) => Some('A'),
            (770, 1209) => Some('4'),
            (770, 1336) => Some('5'),
            (770, 1477) => Some('6'),
            (770, 1633) => Some('B'),
            (852, 1209) => Some('7'),
            (852, 1336) => Some('8'),
            (852, 1477) => Some('9'),
            (852, 1633) => Some('C'),
            (941, 1209) => Some('*'),
            (941, 1336) => Some('0'),
            (941, 1477) => Some('#'),
            (941, 1633) => Some('D'),
            _ => None,
        }
    }
    else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use rodio::{Decoder, Source};

    use super::*;

    #[test]
    pub fn dtmf_test() {
        let test_cases = [
            ("test_audio/Dtmf0.ogg", [941, 1336]),
            ("test_audio/Dtmf1.ogg", [697, 1209]),
            ("test_audio/Dtmf2.ogg", [697, 1336]),
            ("test_audio/Dtmf3.ogg", [697, 1477]),
            ("test_audio/Dtmf4.ogg", [770, 1209]),
            ("test_audio/Dtmf5.ogg", [770, 1336]),
            ("test_audio/Dtmf6.ogg", [770, 1477]),
            ("test_audio/Dtmf7.ogg", [852, 1209]),
            ("test_audio/Dtmf8.ogg", [852, 1336]),
            ("test_audio/Dtmf9.ogg", [852, 1477]),
            ("test_audio/DtmfA.ogg", [697, 1633]),
            ("test_audio/DtmfB.ogg", [770, 1633]),
            ("test_audio/DtmfC.ogg", [852, 1633]),
            ("test_audio/DtmfD.ogg", [941, 1633]),
            ("test_audio/DtmfPound.ogg", [941, 1477]),
            ("test_audio/DtmfStar.ogg", [941, 1209]),
        ];

        for case in test_cases {
            dtmf_single_tone_process(case.0, &case.1);
        }
    }

    fn dtmf_single_tone_process(filename: &str, expected_frequencies: &[u32]) {
        let file = BufReader::new(File::open(filename).unwrap());
        let source = Decoder::new(file).unwrap();
        let sample_rate = source.sample_rate();

        let samples: Vec<f32> = source.convert_samples().collect();
        let minimum_n = sample_rate / 70;

        let mut results: Vec<(u32, f32)> = Vec::new();
        println!("File: {}", filename);
        for freq in DTMF_FREQUENCIES {
            let output = goertzel(freq as f32, sample_rate, &samples[0..minimum_n as usize]);
            println!("{} Hz - {:.0}", freq, output);

            results.push((freq, output));
        }
        let outputs: Vec<f32> = results.iter().map(|(_,output)| *output).collect();
        let mean = outputs.iter().sum::<f32>() / outputs.len() as f32;
        let std_dev: f32 = standard_deviation(&outputs).unwrap();

        let frequency_components: Vec<u32> = results.iter()
            .filter(|(_, output)| *output >= mean + std_dev && *output >= MIN_THRESHOLD )
            .map(|(freq, _)| *freq)
            .collect();
        println!("{:?} - {} - {}", frequency_components, mean, std_dev);

        assert_eq!(
            frequency_components[0] * frequency_components[1], 
            expected_frequencies[0] * expected_frequencies[1],
            "File {}: Expected frequencies {:?} but found {:?}!", filename, expected_frequencies, frequency_components);
    }
}
