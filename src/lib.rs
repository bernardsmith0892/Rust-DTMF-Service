use std::{f32::consts::PI};

const DTMF_FREQUENCIES: [u32; 8] = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
const MIN_THRESHOLD: f32 = 2.0;

/// Object to track timings of a stream to process a multi-digit DTMF signal
#[derive(Debug)]
pub struct DtmfProcessor {
    /// Sample rate of input signal in *samples per second*.
    pub sample_rate: u32,
    /// Number of input channels, assumed to be interleaved.
    pub channels: u16,
    /// The digit we are currently tracking. None if no digit detected.
    candidate: Option<char>,
    /// If we've already returned the candidate digit.
    candidate_pushed: bool,
    /// How long we've been tracking this candidate in *nanoseconds*.
    current_timestamp: u128,
}

impl DtmfProcessor {
    /// Threshold length of a valid DTMF tone in *nanoseconds*.
    const TONE_LENGTH: u128 = 20 * 1_000_000; 
    /// Threshold length of silence in *nanoseconds* to mark a space.
    const SPACE_LENGTH: u128 = 500 * 1_000_000; 

    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self { 
            sample_rate: sample_rate,
            channels: channels, 
            candidate: None,
            candidate_pushed: true,
            current_timestamp: 0,
        }
    }

    pub fn process_samples(&mut self, data: &[f32]) -> Option<char> {
        // Convert multi-channel inputs into mono through averaging
        let input_buffer: Vec<f32> = data.chunks(self.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32 )
            .collect();

        // Increment the current timestamp by the data length
        let nanoseconds_per_sample: u128 = 1_000_000_000 / self.sample_rate as u128;
        self.current_timestamp += input_buffer.len() as u128 * nanoseconds_per_sample;

        // Analyze the input to detect a DTMF signal
        match detect_dtmf_digit(&input_buffer, self.sample_rate) {
            // If we detected a DTMF digit...
            Some(digit) => {
                // Return this digit if:
                //  - It matches the known candidate
                //  - It's been on longer than the minimum tone spacing
                //  - And it's not already pushed to the output buffer
                if Some(digit) == self.candidate {
                    if self.current_timestamp >= DtmfProcessor::TONE_LENGTH
                    && !self.candidate_pushed
                    {
                        // Mark this character as pushed
                        self.candidate_pushed = true;

                        // Print the current char
                        return Some(digit);
                    }
                }
                // If it's a new digit, reset the candidate values
                else {
                    self.candidate = Some(digit);
                    self.current_timestamp = 0;
                    self.candidate_pushed = false;
                }
            },
            // If we do not detect a DTMF signal...
            None => {
                // And we're currently tracking silence
                if self.candidate.is_none() {
                    // Mark a space if there has been enough silence after a previous digit
                    if self.current_timestamp >= DtmfProcessor::SPACE_LENGTH 
                    && !self.candidate_pushed {
                        // Mark this space as pushed
                        self.candidate_pushed = true;     

                        // Print a space
                        return Some(' ');
                    }
                }
                // Start tracking silence if he haven't already
                else {
                    self.candidate = None;
                    self.candidate_pushed = false;
                    self.current_timestamp = 0;
                }
            },
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
/// // Creates a digitized sine wave of a given frequency and sample rate
/// let sine_hz = |n: u32, freq: f32, samp_rate: f32| 
///   f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate);
/// 
/// // Create 0.5 second, 1000 Hz sine wave at 1,024 samples/sec
/// let test_signal: Vec<f32> = (0..512)
///     .map(|n| sine_hz(n, 1000.0, 1024.0))
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
    for (n, &sample) in samples.iter().enumerate() {
        // Hamming Window
        let adjusted_sample = sample * (0.5 - 0.25 * f32::cos(2.0 * PI * (n as f32 / samples.len() as f32)));
        q.0 = adjusted_sample + coefficient * q.1 - q.2;
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

/// Returns a vector of matching DTMF frequencies for this signal.
/// 
/// A frequency is considered matching if its power is higher than the minimum threshold value (2.0) and it is at least one standard deviation stronger than the average power of all eight DTMF frequencies.
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
/// // Creates a digitized sine wave of a given frequency and sample rate
/// let sine_hz = |n: u32, freq: f32, samp_rate: f32| 
///   f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate);
/// 
/// // Create 0.5 second signal with 697 Hz and 1633 Hz components (DMTF `A`)
/// let test_signal: Vec<f32> = (0..512)
///     .map(|n| sine_hz(n, 697.0, 1024.0) + sine_hz(n, 1633.0, 1024.0))
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
/// // Creates a digitized sine wave of a given frequency and sample rate
/// let sine_hz = |n: u32, freq: f32, samp_rate: f32| 
///   f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate);
/// 
/// // Create 0.5 second signal with 697 Hz and 1633 Hz components (DMTF `A`)
/// let a_signal: Vec<f32> = (0..512)
///     .map(|n| sine_hz(n, 697.0, 1024.0) + sine_hz(n, 1633.0, 1024.0))
///     .collect();
///
/// // Create 0.5 second signal with 1000 Hz and 600 Hz components (invalid DTMF)
/// let bad_signal: Vec<f32> = (0..512)
///     .map(|n| sine_hz(n, 1000.0, 1024.0) + sine_hz(n, 600.0, 1024.0))
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
    pub fn dtmf_single_digit_test() {
        let test_cases = [
            ("test_audio/Dtmf0.ogg", '0'),
            ("test_audio/Dtmf1.ogg", '1'),
            ("test_audio/Dtmf2.ogg", '2'),
            ("test_audio/Dtmf3.ogg", '3'),
            ("test_audio/Dtmf4.ogg", '4'),
            ("test_audio/Dtmf5.ogg", '5'),
            ("test_audio/Dtmf6.ogg", '6'),
            ("test_audio/Dtmf7.ogg", '7'),
            ("test_audio/Dtmf8.ogg", '8'),
            ("test_audio/Dtmf9.ogg", '9'),
            ("test_audio/DtmfA.ogg", 'A'),
            ("test_audio/DtmfB.ogg", 'B'),
            ("test_audio/DtmfC.ogg", 'C'),
            ("test_audio/DtmfD.ogg", 'D'),
            ("test_audio/DtmfPound.ogg", '#'),
            ("test_audio/DtmfStar.ogg", '*'),
        ];

        for case in test_cases {
            dtmf_single_tone_process(case.0, case.1);
        }
    }

    fn dtmf_single_tone_process(filename: &str, expected_digit: char) {
        let file = BufReader::new(File::open(filename).unwrap());
        let source = Decoder::new(file).unwrap();
        let sample_rate = source.sample_rate();

        let samples: Vec<f32> = source.convert_samples().collect();

        let digit = detect_dtmf_digit(&samples[0..480], sample_rate).unwrap();

        assert_eq!(digit, expected_digit,
            "File {}: Expected {} but found {}!", filename, expected_digit, digit);
    }

    #[test]
    fn dtmf_multiple_digits_test() {
        let file = BufReader::new(File::open("test_audio/DTMF_dialing.ogg").unwrap());
        let source = Decoder::new(file).unwrap();
        let sample_rate = source.sample_rate();
        let channels = source.channels();
        let samples: Vec<f32> = source.convert_samples().collect();

        let mut processor = DtmfProcessor::new(sample_rate, channels);

        println!("{:?}", processor);
        let mut digits = String::new();

        // Chunks of 10ms
        for block in samples.chunks(sample_rate as usize / 100) { 
            if let Some(digit) = processor.process_samples(block) {
                digits.push(digit);
            }
        }

        assert_eq!(digits, "06966753564646415180233673141636083381604400826146625368963884821381785073643399");
    }
}
