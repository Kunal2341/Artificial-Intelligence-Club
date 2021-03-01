var arDrone = require('ar-drone');
var client  = arDrone.createClient();
client.createRepl();

/*
client.takeoff();
client.after(3000, function() {
    this.stop();
    this.land();
  });
//This flies for 3 seconds and then turns of the motors, maybe crash?
*/
var pngStream = client.getPngStream();

