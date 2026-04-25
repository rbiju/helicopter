#include <IRremote.hpp>

#define IR_PIN    3
#define READY_ACK 129

uint32_t reverseBits(uint32_t n) {
  uint32_t res = 0;
  for (int i = 0; i < 32; i++) {
    res = (res << 1) | (n & 1);
    n >>= 1;
  }
  return res;
}

void setup() {
  Serial.begin(115200);
  IrReceiver.begin(IR_PIN, ENABLE_LED_FEEDBACK);
}

void loop() {
  if (IrReceiver.decode()) {
    uint32_t rawData = IrReceiver.decodedIRData.decodedRawData;

    if (rawData != 0) {
      uint32_t reversedData = reverseBits(rawData);

      byte data0 = (reversedData >> 24) & 0xFF;
      byte data1 = (reversedData >> 16) & 0xFF;
      byte data2 = (reversedData >> 8) & 0xFF;
      byte data3 = reversedData & 0xFF;

      byte trim_val = data3;
      byte throttle = data2 & 127;
      char channel = (data2 & 128) ? 'B' : 'A';
      byte pitch = data1;
      byte yaw = data0 - (int(trim_val) - 63) / 3;

      Serial.print("CH:"); Serial.print(channel);
      Serial.print(" YAW:"); Serial.print(yaw);
      Serial.print(" PITCH:"); Serial.print(pitch);
      Serial.print(" THR:"); Serial.print(throttle);
      Serial.print(" TRIM:"); Serial.println(trim_val);

//       Serial.write(READY_ACK)
//       Serial.write(channel);
//       Serial.write(yaw);
//       Serial.write(pitch);
//       Serial.write(throttle);
//       Serial.write(trim_val);
    }

    IrReceiver.resume();
  }
}