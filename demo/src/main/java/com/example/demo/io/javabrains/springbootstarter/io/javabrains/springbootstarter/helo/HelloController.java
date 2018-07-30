package com.example.demo.io.javabrains.springbootstarter.io.javabrains.springbootstarter.helo;




import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController // this class is a controller, use the info and where it's coming from
public class HelloController {

    @RequestMapping("/hello")  //maps all http requests to methods
    public String Hi(){

        return "hi";
    }

}