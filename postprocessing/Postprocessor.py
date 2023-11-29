from pydub import AudioSegment
import numpy as np
import librosa
from scipy.signal import butter, lfilter

class Postprocessor:
    def remove_noise(self, signal, noise_window_size=500, threshold=500):
        noise = np.abs(signal[:noise_window_size])
        is_noise = np.abs(signal) < threshold
        signal[is_noise] = 0
        return signal


    def high_pass_filter(self, signal, cutoff_frequency=500, sampling_rate=44100):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, signal)


    def apply_dynamic_range_compression(self, audio, threshold_db=-20.0, ratio=4.0):
        # Convert to numpy array
        samples = audio.get_array_of_samples()

        # Apply compression
        compressed_samples = [max(min(int(sample * ratio), 32767), -32768) for sample in samples]

        # Convert back to AudioSegment
        compressed_audio = AudioSegment(np.int16(compressed_samples).tobytes(),
                                        frame_rate=audio.frame_rate,
                                        sample_width=audio.sample_width,
                                        channels=audio.channels)
        return compressed_audio


    def time_stretch(self, audio, speed_factor=1.0):
        # Adjust duration using speedup/slow down
        stretched_audio = audio.speedup(playback_speed=speed_factor)
        return stretched_audio


    def remove_reverb(self, audio, delay_ms=50, attenuation_factor=0.5):
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Create a delayed version of the signal
        delayed_samples = np.roll(samples, int(audio.frame_rate * delay_ms / 1000))

        # Subtract the delayed version with an attenuation factor
        processed_samples = samples - attenuation_factor * delayed_samples

        # Ensure that the values are within the valid range
        processed_samples = np.clip(processed_samples, -32768, 32767)

        # Convert back to AudioSegment
        processed_audio = AudioSegment(np.int16(processed_samples).tobytes(),
                                       frame_rate=audio.frame_rate,
                                       sample_width=audio.sample_width,
                                       channels=audio.channels)
        return processed_audio


    def de_ess(self, audio, threshold_db=-20.0, reduction_factor=0.5):
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Compute the energy in the high-frequency range (e.g., above 4 kHz)
        high_freq_energy = np.sum(samples[samples > threshold_db])

        # Compute the reduction gain based on the energy in the high-frequency range
        reduction_gain = -reduction_factor * high_freq_energy / len(samples)

        # Apply the reduction gain to the high-frequency components
        processed_samples = samples + reduction_gain * samples

        # Ensure that the values are within the valid range
        processed_samples = np.clip(processed_samples, -32768, 32767)

        # Convert back to AudioSegment
        processed_audio = AudioSegment(np.int16(processed_samples).tobytes(),
                                       frame_rate=audio.frame_rate,
                                       sample_width=audio.sample_width,
                                       channels=audio.channels)
        return processed_audio


    def adjust_volume_envelope(self, audio, attack_time=100, decay_time=100, sustain_level=0.8, release_time=100):
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Create an envelope with attack, decay, sustain, and release phases
        envelope = np.ones(len(samples))
        envelope[:attack_time] = np.linspace(0, 1, attack_time)
        envelope[attack_time:attack_time + decay_time] = np.linspace(1, sustain_level, decay_time)
        envelope[-release_time:] = np.linspace(sustain_level, 0, release_time)

        # Apply the envelope to the audio samples
        processed_samples = samples * envelope

        # Ensure that the values are within the valid range
        processed_samples = np.clip(processed_samples, -32768, 32767)

        # Convert back to AudioSegment
        processed_audio = AudioSegment(np.int16(processed_samples).tobytes(),
                                       frame_rate=audio.frame_rate,
                                       sample_width=audio.sample_width,
                                       channels=audio.channels)
        return processed_audio


    def pitch_correct(self, audio, factor=1.0):
        # Adjust pitch using speedup/slow down
        pitch_corrected_audio = audio.speedup(playback_speed=factor)
        return pitch_corrected_audio


    def apply_limiter(self, audio, threshold_db=-3.0):
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Calculate the peak amplitude
        peak_amplitude = np.max(np.abs(samples))

        # Calculate the gain to apply to the samples to limit the amplitude
        limiting_gain = 10 ** ((threshold_db - 20 * np.log10(peak_amplitude)) / 20)

        # Apply the limiting gain to the samples
        limited_samples = limiting_gain * samples

        # Ensure that the values are within the valid range
        limited_samples = np.clip(limited_samples, -32768, 32767)

        # Convert back to AudioSegment
        limited_audio = AudioSegment(np.int16(limited_samples).tobytes(),
                                     frame_rate=audio.frame_rate,
                                     sample_width=audio.sample_width,
                                     channels=audio.channels)
        return limited_audio


    def process(self, path):
        # Load audio file
        audio_path = path
        audio = AudioSegment.from_file(audio_path, format="wav")
        samples = np.array(audio.get_array_of_samples())

        samples = self.remove_noise(samples)
        samples = self.high_pass_filter(samples)
        samples = self.pitch_correct(samples)
        samples = self.de_ess(samples)

        return samples
