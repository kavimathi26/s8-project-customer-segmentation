import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Route, Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  constructor(private formBuilder:FormBuilder,private router:Router ){

  }
 
  profileForm:FormGroup;
  ngOnInit(){
    this.profileForm = this.formBuilder.group({
     Username:['',[Validators.required]],
     Password:['',[Validators.required]]
   });
 
   
  }
 
  saveForm(){
    if(this.profileForm.valid){
      console.log('Profile form data :: ', this.profileForm.value)
      this.router.navigate(['/dashboard'])
    }
  }
 
 }
 