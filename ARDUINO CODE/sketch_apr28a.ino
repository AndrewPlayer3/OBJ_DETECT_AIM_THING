void setup() {
  Serial.begin(57600);
  pinMode(13, OUTPUT);
}

void loop() {
  String input = Serial.readString();
  Serial.print("I recieved: ");
  Serial.println(input);
  digitalWrite(13, HIGH);
  if(input == "SHOOT") {
    //If we had the nerf gun and/or servos the movement/shoot code would go here.
    //For now, it just shuts off the LED on the arduino
    digitalWrite(13, LOW);
    Serial.println("GOT IT");
    delay(10);
  }
}
