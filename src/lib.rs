use std::{f32::consts::PI, collections::HashMap, sync::Mutex};
use cpal::{StreamInstant};
use lazy_static::lazy_static;

const DTMF_FREQUENCIES: [u32; 8] = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
const MIN_THRESHOLD: f32 = 2.0;
lazy_static! {
static ref DTMF_MAP: HashMap<u32, char> = HashMap::from(
    [
        (697 * 1209, '1'),
        (697 * 1336, '2'),
        (697 * 1477, '3'),
        (697 * 1633, 'A'),
        (770 * 1209, '4'),
        (770 * 1336, '5'),
        (770 * 1477, '6'),
        (770 * 1633, 'B'),
        (852 * 1209, '7'),
        (852 * 1336, '8'),
        (852 * 1477, '9'),
        (852 * 1633, 'C'),
        (941 * 1209, '*'),
        (941 * 1336, '0'),
        (941 * 1477, '#'),
        (941 * 1633, 'D'),
    ]
);
}

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
            // Convert multi-channel inputs into mono by averaging
            data.chunks(self.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32 )
            .collect()
        );

        // Process the current input if we have enough samples
        if input_buffer.len() >= self.minimum_n as usize {
            // Analyze the input to detect a DTMF signal
            let current_tone = detect_dtmf(&input_buffer, self.sample_rate);
            let current_time = info.timestamp().capture;

            match current_tone {
                // If we detected a DTMF signal
                Some(&tone) => match self.candidate {
                    // Add this digit if:
                    //  - It matches the candidate
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
                    // Change the candidate if it's different from the current tone
                    _ => self.candidate = (tone, Some(current_time), false),
                },
                // If we do not detect any DTMF tones...
                None => {
                    match self.candidate {
                        // Mark a space if there has been enough silence
                        (' ', Some(silence_start), pushed) => {
                            if current_time.duration_since(&silence_start)
                                .unwrap().as_millis() >= SPACE_LENGTH 
                            && !pushed {
                                // Mark this space as pushed
                                self.candidate = (' ', Some(silence_start), true);     

                                // Print the space and clear the input buffer
                                input_buffer.clear();
                                return Some(' ');
                            }
                        },
                        // The base case - Add a timestamp and do not push a leading space char
                        (' ', None, pushed) if pushed => self.candidate = (' ', Some(current_time), true),

                        // Start marking silence if don't detect a character
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

pub fn standard_deviation(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        None
    }
    else {
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 = data.iter().map(|value| (mean - value).powf(2.0)).sum::<f32>() / data.len() as f32;

        Some(variance.sqrt())
    }
}

pub fn get_dtmf_components(samples: &[f32], sample_rate: u32) -> Option<Vec<u32>> {
    let results: Vec<f32> = 
        DTMF_FREQUENCIES.iter()
        .map(|freq| goertzel(*freq as f32, sample_rate, samples))
        .collect();
    let mean = results.iter().sum::<f32>() / results.len() as f32;
    let std_dev: f32 = standard_deviation(&results).unwrap();

    let frequency_components: Vec<u32> = 
    results.into_iter()
        .zip(DTMF_FREQUENCIES)
        .filter(|(output, _)| *output >= mean + std_dev && *output >= MIN_THRESHOLD)
        .map(|(_, freq)| freq)
        .collect();

    match frequency_components.is_empty() {
        true => None,
        false => Some(frequency_components),
    }
}

pub fn detect_dtmf(samples: &[f32], sample_rate: u32) -> Option<&char> {
    match get_dtmf_components(samples, sample_rate) {
        Some(freqs) if freqs.len() >= 2 => DTMF_MAP.get(&(freqs[0] * freqs[1])), 
        _ => None,
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
