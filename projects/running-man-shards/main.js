var gaussApprox = function (mu, sigma) {
    return ((Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random()) - 3) * sigma + mu;
}

var generatePolygon = function (ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts) {
    irregularity = Math.max(0, Math.min(irregularity, 1)) * 2 * Math.PI / numVerts;
    spikeyness = Math.max(0, Math.min(spikeyness, 1)) * aveRadius;

    var angleSteps = [];
    var lower = (2 * Math.PI / numVerts) - irregularity;
    var upper = (2 * Math.PI / numVerts) + irregularity;
    var sum = 0;
    for (var i = 0; i < numVerts; i++) {
        var tmp = Math.random() * (upper - lower) + lower;
        angleSteps.push(tmp);
        sum += tmp;
    }

    var k = sum / (2 * Math.PI)
    for (var i = 0; i < numVerts; i++) {
        angleSteps[i] = angleSteps[i] / k;
    }

    var points = [];
    var angle = Math.random() * 2 * Math.PI;
    for (var i = 0; i < numVerts; i++) {
        var r_i = Math.max(0, Math.min(gaussApprox(aveRadius, spikeyness), 2 * aveRadius));
        var x = ctrX + r_i * Math.cos(angle);
        var y = ctrY + r_i * Math.sin(angle);
        points.push(x);
        points.push(y);
        angle += angleSteps[i];
    }

    var triangulatedPoints = [];
    for (var i = 0; i < (numVerts - 2); i++) {
        triangulatedPoints.push(points[0]);
        triangulatedPoints.push(points[1]);
        triangulatedPoints.push(points[2 * i + 2]);
        triangulatedPoints.push(points[2 * i + 3]);
        triangulatedPoints.push(points[2 * i + 4]);
        triangulatedPoints.push(points[2 * i + 5]);
    }

    return triangulatedPoints;
}

var Shard = function (n) {
    this.creationTime = parameters.time;
    this.verts = generatePolygon(0.0, 0.0, 1.0, 0.4, 1.0, n);

    // Create Vertex buffer

    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array (this.verts), gl.STATIC_DRAW);
    totalVerts += this.verts.length / 2;

    this.modelMat = mat4.create();
    mat4.translate(this.modelMat, this.modelMat, [Math.random()*6.0 - 3.0, 5.0, Math.random() * (0.0 - (-8.0)) + (-8.0)]);
    this.rotationSpeed = Math.random() * (0.2 - 0.02) + 0.02;
    this.rotationAxes = [Math.random() * 2.0 - 1.0, Math.random() * 2.0 - 1.0, Math.random() * 2.0 - 1.0];
    this.fallSpeed = 1.5;
}

/**
 * Provides requestAnimationFrame in a cross browser way.
 * paulirish.com/2011/requestanimationframe-for-smart-animating/
*/
window.requestAnimationFrame = window.requestAnimationFrame || ( function() {
    return  window.webkitRequestAnimationFrame ||
            window.mozRequestAnimationFrame ||
            window.oRequestAnimationFrame ||
            window.msRequestAnimationFrame ||
            function(  callback, element ) {
                window.setTimeout( callback, 1000 / 60 );
            };
})();

var canvas,
    gl,
    outputRectBuf,
    shardProgram, hBlurProgram, vBlurProgram,
    vertex_position, output_vertex_position,
    totalVerts,
    shards,
    parameters = { start_time   : new Date().getTime(),
                   time         : 0,
                   lastShardTime: 0,
                   screenWidth  : 0,
                   screenHeight : 0 };

init();
animate();

function init() {

    var vertex_shader = document.getElementById('vs').textContent;
    var fragment_shader = document.getElementById('fs').textContent;
    var output_vertex_shader = document.getElementById('output-vs').textContent;
    var hBlurShader = document.getElementById('hblur-fs').textContent;
    var vBlurShader = document.getElementById('vblur-fs').textContent;

    totalVerts = 0;
    shards = [];

    canvas = document.querySelector( 'canvas' );

    // Initialise WebGL

    try {

        gl = canvas.getContext( 'experimental-webgl' );

    } catch( error ) { }

    if ( !gl ) {

        throw "cannot create webgl context";

    }

    // Some initial shards

    shards.push(new Shard(5));
    shards.push(new Shard(5));
    shards.push(new Shard(4));
    shards.push(new Shard(4));

    // Create output rectangle

    var outputRectVertices = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    outputRectBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, outputRectBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array (outputRectVertices), gl.STATIC_DRAW);

    // Create Programs

    shardProgram = createProgram( vertex_shader, fragment_shader );
    hBlurProgram = createProgram( output_vertex_shader, hBlurShader );
    vBlurProgram = createProgram( output_vertex_shader, vBlurShader );

    // Set attrib locations
    vertex_position = gl.getAttribLocation(shardProgram, 'position');
    hBlurPositionLocation = gl.getAttribLocation(hBlurProgram, 'position');
    vBlurPositionLocation = gl.getAttribLocation(vBlurProgram, 'position');

    onWindowResize();
    window.addEventListener( 'resize', onWindowResize, false );

}

function createProgram( vertex, fragment ) {

    var program = gl.createProgram();

    var vs = createShader( vertex, gl.VERTEX_SHADER );
    var fs = createShader( '#ifdef GL_ES\nprecision highp float;\n#endif\n\n' + fragment, gl.FRAGMENT_SHADER );

    if ( vs == null || fs == null ) return null;

    gl.attachShader( program, vs );
    gl.attachShader( program, fs );

    gl.deleteShader( vs );
    gl.deleteShader( fs );

    gl.linkProgram( program );

    if ( !gl.getProgramParameter( program, gl.LINK_STATUS ) ) {

        alert( "ERROR:\n" +
        "VALIDATE_STATUS: " + gl.getProgramParameter( program, gl.VALIDATE_STATUS ) + "\n" +
        "ERROR: " + gl.getError() + "\n\n" +
        "- Vertex Shader -\n" + vertex + "\n\n" +
        "- Fragment Shader -\n" + fragment );

        return null;

    }

    return program;

}

function createShader( src, type ) {

    var shader = gl.createShader( type );

    gl.shaderSource( shader, src );
    gl.compileShader( shader );

    if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {

        alert( ( type == gl.VERTEX_SHADER ? "VERTEX" : "FRAGMENT" ) + " SHADER:\n" + gl.getShaderInfoLog( shader ) );
        return null;

    }

    return shader;

}

function onWindowResize( event ) {

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    parameters.screenWidth = canvas.width;
    parameters.screenHeight = canvas.height;

}

function animate() {

    requestAnimationFrame( animate );
    render();

}

function render() {

    if ( !shardProgram ) return;

    parameters.time = new Date().getTime() - parameters.start_time;

    if (parameters.time - parameters.lastShardTime > 1000) {
        parameters.lastShardTime = parameters.time;
        shards.push(new Shard(Math.random() > 0.75 ? 5 : 4));
    }

    gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );

    var downsample = 0.5;

    // Generate FBO and Texture for offscreen render

    var downsampledShardsFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, downsampledShardsFbo);

    var downsampledShardsTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, downsampledShardsTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, parameters.screenWidth * downsample, parameters.screenHeight * downsample, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, downsampledShardsTex, 0);

    var fullShardsFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fullShardsFbo);

    var fullShardsTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, fullShardsTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, parameters.screenWidth, parameters.screenHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fullShardsTex, 0);

    gl.useProgram( shardProgram );

    var shardsToDelete = [];
    for (var i = 0; i < shards.length; i++) {
        if (parameters.time - shards[i].creationTime > 30000.0) {
            shardsToDelete.push(i);
            continue;
        }

        // Set values to program variables

        var model = mat4.clone(shards[i].modelMat);
        mat4.scale(model, model, [0.1, 0.25, 0.25]);
        mat4.translate(model, model, [0.0, -shards[i].fallSpeed * (parameters.time - shards[i].creationTime) / 1000.0, 0.0]);
        mat4.rotate(model, model, shards[i].rotationSpeed * 2.0 * Math.PI * (parameters.time - shards[i].creationTime) / 1000.0, shards[i].rotationAxes);

        var view = mat4.create();
        mat4.identity(view);
        mat4.lookAt(view, [0.0, 0.0, 3.0], [0.0,0.0,0.0], [0.0, 1.0, 0.0]);


        var persp = mat4.create();
        mat4.perspective(persp, Math.PI * 45.0 / 180.0, parameters.screenWidth / parameters.screenHeight, 1.0, 10.0);

        var mvp = mat4.create();
        mat4.multiply(mvp, persp, mat4.multiply(mvp, view, model));

        gl.uniform1f( gl.getUniformLocation( shardProgram, 'time' ), parameters.time / 1000 );
        gl.uniform2f( gl.getUniformLocation( shardProgram, 'resolution' ), parameters.screenWidth, parameters.screenHeight );
        gl.uniformMatrix4fv( gl.getUniformLocation( shardProgram, 'mvp' ), gl.False, mvp );

        // Render geometry

        gl.bindBuffer( gl.ARRAY_BUFFER, shards[i].buffer );
        gl.vertexAttribPointer( vertex_position, 2, gl.FLOAT, false, 0, 0 );
        gl.enableVertexAttribArray( vertex_position );

        gl.bindFramebuffer( gl.FRAMEBUFFER, downsampledShardsFbo );
        gl.viewport( 0, 0, parameters.screenWidth * downsample, parameters.screenHeight * downsample );
        gl.drawArrays( gl.TRIANGLES, 0, shards[i].verts.length / 2 );

        gl.bindFramebuffer( gl.FRAMEBUFFER, fullShardsFbo );
        gl.viewport( 0, 0, parameters.screenWidth, parameters.screenHeight );
        gl.drawArrays( gl.TRIANGLES, 0, shards[i].verts.length / 2 );

        gl.disableVertexAttribArray( vertex_position );
    }

    for (var i = shardsToDelete.length; i > 0; i--) {
        shards.splice(shardsToDelete[i - 1], 1);
    }
    //console.log(shards.length);

    // Gen fbo and tex for storing temp horizontal blur

    var tempBlurFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, tempBlurFbo);

    var tempBlurTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, tempBlurTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, parameters.screenWidth * downsample, parameters.screenHeight * downsample, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tempBlurTex, 0);

    // Render hblur

    gl.useProgram(hBlurProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, tempBlurFbo);
    gl.uniform1i(gl.getUniformLocation(hBlurProgram, 'shardTex'), 0);
    gl.uniform2f(gl.getUniformLocation(hBlurProgram, 'resolution'), parameters.screenWidth * downsample, parameters.screenHeight * downsample);
    gl.bindBuffer(gl.ARRAY_BUFFER, outputRectBuf);
    gl.viewport( 0, 0, parameters.screenWidth * downsample, parameters.screenHeight * downsample );
    gl.vertexAttribPointer( hBlurPositionLocation, 2, gl.FLOAT, false, 0, 0 );
    gl.enableVertexAttribArray( hBlurPositionLocation );
    gl.drawArrays( gl.TRIANGLES, 0, 6 );
    gl.disableVertexAttribArray( hBlurPositionLocation );

    // Render vblur (to screen)

    gl.useProgram(vBlurProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.uniform1i(gl.getUniformLocation(vBlurProgram, 'fullShardTex'), 1);
    gl.uniform1i(gl.getUniformLocation(vBlurProgram, 'hblurTex'), 2);
    gl.uniform2f(gl.getUniformLocation(vBlurProgram, 'resolution'), parameters.screenWidth * downsample, parameters.screenHeight * downsample);
    gl.bindBuffer(gl.ARRAY_BUFFER, outputRectBuf);
    gl.viewport( 0, 0, parameters.screenWidth, parameters.screenHeight );
    gl.vertexAttribPointer( vBlurPositionLocation, 2, gl.FLOAT, false, 0, 0 );
    gl.enableVertexAttribArray( vBlurPositionLocation );
    gl.drawArrays( gl.TRIANGLES, 0, 6 );
    gl.disableVertexAttribArray( vBlurPositionLocation );

}
