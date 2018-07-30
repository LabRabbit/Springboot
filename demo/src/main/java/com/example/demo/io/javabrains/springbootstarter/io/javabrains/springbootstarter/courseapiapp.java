package com.example.demo.io.javabrains.springbootstarter.io.javabrains.springbootstarter;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;

@SpringBootApplication
@EnableAutoConfiguration(exclude={DataSourceAutoConfiguration.class}) //to disable datasource
public class courseapiapp{
    public static void main(String args[]){
        SpringApplication.run(courseapiapp.class,args);
    }

}

