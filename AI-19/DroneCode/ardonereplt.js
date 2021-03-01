var arDrone    = require('ar-drone'),
	fs         = require('fs'),
	dateFormat = require('dateformat');
var client    = arDrone.createClient();
var pngStream = client.getPngStream(); // 1

function takePhoto() {
    pngStream.once('data', function (data) { // 2
        var now = new Date();
        var nowFormat = dateFormat(now, 'isoDateTime');
        
        // 3
        fs.writeFile('image-' + nowFormat + '.png', data, function (err) {
            if (err)
                console.error(err);
            else
                console.log('Photo saved');
        })
    });
}

takePhoto();